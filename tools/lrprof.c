// lrprof.c - a tiny sampling profiler for Windows (docs/PERF_REVIEW.md), linked into a
// program when it is to be profiled; nothing of the engine includes it.
//
//   gcc ... apps/bench.c tools/lrprof.c build/float-vk/liblr.a ... -o out/perf/bench_prof.exe
//   set LRPROF=out/perf/b2.prof     (where the samples go; without it nothing runs)
//   set LRPROF_HZ=2000              (samples a second, default 1000)
//   out/perf/bench_prof.exe -only B2
//   node tools/lrprof.mjs out/perf/bench_prof.exe out/perf/b2.prof
//
// A thread of its own wakes LRPROF_HZ times a second, stops every other thread of the
// process for a moment, reads its instruction pointer and lets it go. Threads that did not
// run since the last sample (no CPU cycles spent: waiting, asleep) are left out, so the
// samples are CPU time. The samples (address, thread) are written when the program ends;
// tools/lrprof.mjs turns them into functions and lines with addr2line (the program needs
// -g). Between stopping a thread and letting it go the profiler calls nothing that takes a
// lock (the stopped thread may hold the heap's).
#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_THREADS 256
#define MAX_SAMPLES (8u << 20)

typedef struct { DWORD id; HANDLE h; ULONG64 cycles; } Th;
static Th th[MAX_THREADS];
static int nth;
static ULONG64 *samples;         // pairs: the address (the thread's slot in its top 8 bits), and for an
                                 // address outside the program the first frame of its stack inside it (or 0)
static ULONG64 exe_lo, exe_hi;
static volatile LONG nsamples;
static volatile LONG stop;
static HANDLE prof_thread;
static char out_path[512];
static int hz = 1000;
static ULONG64 ticks, min_cycles;

static void threads_refresh(DWORD self){
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return;
    THREADENTRY32 te = {.dwSize = sizeof(te)};
    DWORD pid = GetCurrentProcessId();
    if (Thread32First(snap, &te)) do {
        if (te.th32OwnerProcessID != pid || te.th32ThreadID == self) continue;
        int known = 0;
        for (int i = 0; i < nth && !known; i++) known = th[i].id == te.th32ThreadID;
        if (known || nth >= MAX_THREADS) continue;
        HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, te.th32ThreadID);
        if (!h) continue;
        th[nth].id = te.th32ThreadID; th[nth].h = h; th[nth].cycles = 0; nth++;
    } while (Thread32Next(snap, &te));
    CloseHandle(snap);
}

// ---- who called into a system library or the C runtime ----
// The stack unwound (the tables every 64-bit Windows module carries) up to the first frame
// inside the program. The system's own lookup of those tables takes a lock the stopped
// thread may hold (while a library loads), so the modules' tables are listed beforehand
// (modules_refresh, with no thread stopped) and searched here; RtlVirtualUnwind itself only
// reads memory. A frame in a module not listed yet ends the walk (no caller).
typedef struct { ULONG64 lo, hi; const RUNTIME_FUNCTION *rf; DWORD n; } Mod;
#define MAX_MODS 256
static Mod modtab[MAX_MODS];
static volatile LONG nmods;

static void modules_refresh(void){
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, 0);
    if (snap == INVALID_HANDLE_VALUE) return;
    MODULEENTRY32 me = {.dwSize = sizeof(me)};
    if (Module32First(snap, &me)) do {
        ULONG64 lo = (ULONG64)(ULONG_PTR)me.modBaseAddr;
        int known = 0;
        for (LONG i = 0; i < nmods && !known; i++) known = modtab[i].lo == lo;
        if (known || nmods >= MAX_MODS) continue;
        const IMAGE_DOS_HEADER *dos = (const IMAGE_DOS_HEADER *)me.modBaseAddr;
        const IMAGE_NT_HEADERS *nt = (const IMAGE_NT_HEADERS *)((const char *)dos + dos->e_lfanew);
        const IMAGE_DATA_DIRECTORY *d = &nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
        Mod *m = &modtab[nmods];
        m->lo = lo; m->hi = lo + me.modBaseSize;
        m->rf = d->Size ? (const RUNTIME_FUNCTION *)((const char *)dos + d->VirtualAddress) : NULL;
        m->n = d->Size / sizeof(RUNTIME_FUNCTION);
        nmods++;
    } while (Module32Next(snap, &me));
    CloseHandle(snap);
}

static ULONG64 caller_in_exe(CONTEXT *c){
    if (c->Rip >= exe_lo && c->Rip < exe_hi) return 0;
    for (int depth = 0; depth < 64; depth++){
        const Mod *m = NULL;
        for (LONG i = 0; i < nmods && !m; i++) if (c->Rip >= modtab[i].lo && c->Rip < modtab[i].hi) m = &modtab[i];
        if (!m) return 0;
        DWORD rva = (DWORD)(c->Rip - m->lo);
        const RUNTIME_FUNCTION *f = NULL;
        for (DWORD lo = 0, hi = m->n; lo < hi; ){
            DWORD mid = (lo + hi) / 2;
            if (rva < m->rf[mid].BeginAddress) hi = mid;
            else if (rva >= m->rf[mid].EndAddress) lo = mid + 1;
            else { f = &m->rf[mid]; break; }
        }
        if (f){
            ULONG64 establisher = 0;
            PVOID handler = NULL;
            RtlVirtualUnwind(0, m->lo, c->Rip, (PRUNTIME_FUNCTION)f, c, &handler, &establisher, NULL);
        } else {                                         // a leaf: the return address is on top
            if (c->Rsp < 0x10000 || (c->Rsp & 7)) return 0;
            c->Rip = *(const ULONG64 *)(ULONG_PTR)c->Rsp;
            c->Rsp += 8;
        }
        if (c->Rip < 0x10000) return 0;
        if (c->Rip >= exe_lo && c->Rip < exe_hi) return c->Rip - 1;   // (inside the call instruction)
    }
    return 0;
}

static DWORD WINAPI prof_main(LPVOID arg){
    (void)arg;
    DWORD self = GetCurrentThreadId();
    HANDLE timer = CreateWaitableTimerExW(NULL, NULL, 0x00000002 /* high resolution */, TIMER_ALL_ACCESS);
    LARGE_INTEGER due;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    {   // the cycles of a 16th of a period, from this thread's own cycle counter over 20 ms
        ULONG64 c0 = 0, c1 = 0;
        LARGE_INTEGER f, q0, q1;
        QueryPerformanceFrequency(&f);
        QueryPerformanceCounter(&q0);
        QueryThreadCycleTime(GetCurrentThread(), &c0);
        do QueryPerformanceCounter(&q1); while ((q1.QuadPart - q0.QuadPart) * 1000 < f.QuadPart * 20);
        QueryThreadCycleTime(GetCurrentThread(), &c1);
        min_cycles = (c1 - c0) * 50 / (ULONG64)hz / 16;
    }
    while (!stop){
        if (ticks % 64 == 0){ threads_refresh(self); modules_refresh(); }
        ticks++;
        if (timer){ due.QuadPart = -(10000000LL / hz); SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE); WaitForSingleObject(timer, 100); }
        else Sleep(1);
        for (int i = 0; i < nth; i++){
            ULONG64 c = 0;
            if (!QueryThreadCycleTime(th[i].h, &c)) continue;
            // did it run since the last sample? (Being stopped and let go costs a waiting
            // thread a few thousand cycles of its own: under a 16th of the period is "no")
            ULONG64 ran = c - th[i].cycles;
            th[i].cycles = c;
            if (ran < min_cycles) continue;
            if (SuspendThread(th[i].h) == (DWORD)-1) continue;
            CONTEXT ctx;
            ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
            if (GetThreadContext(th[i].h, &ctx) && nsamples < (LONG)MAX_SAMPLES){
                samples[2 * nsamples] = ((ULONG64)i << 56) | (ctx.Rip & 0x00ffffffffffffffULL);
                samples[2 * nsamples + 1] = caller_in_exe(&ctx);
                nsamples++;
            }
            ResumeThread(th[i].h);
        }
    }
    return 0;
}

static void prof_dump(void){
    stop = 1;
    if (prof_thread) WaitForSingleObject(prof_thread, 2000);
    FILE *f = fopen(out_path, "w");
    if (!f) return;
    HMODULE exe = GetModuleHandleA(NULL);
    fprintf(f, "base %llx hz %d ticks %llu samples %ld threads %d\n", (unsigned long long)(ULONG_PTR)exe, hz, (unsigned long long)ticks, (long)nsamples, nth);
    // the modules the samples fall into
    HMODULE mods[64]; int nm = 0;
    for (LONG i = 0; i < nsamples; i++){
        ULONG64 a = samples[2 * i] & 0x00ffffffffffffffULL, from = samples[2 * i + 1];
        HMODULE m = NULL;
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)(ULONG_PTR)a, &m);
        int k = 0;
        while (k < nm && mods[k] != m) k++;
        if (k == nm && nm < 64){
            mods[nm++] = m;
            char name[512] = "?";
            if (m) GetModuleFileNameA(m, name, sizeof(name));
            fprintf(f, "module %d %llx %s\n", k, (unsigned long long)(ULONG_PTR)m, name);
        }
        fprintf(f, "s %d %d %llx %llx\n", (int)(samples[2 * i] >> 56), k, (unsigned long long)(a - (ULONG64)(ULONG_PTR)m),
                (unsigned long long)(from ? from - exe_lo : 0));
    }
    fclose(f);
}

__attribute__((constructor)) static void prof_init(void){
    const char *o = getenv("LRPROF");
    if (!o || !*o) return;
    snprintf(out_path, sizeof(out_path), "%s", o);
    const char *h = getenv("LRPROF_HZ");
    if (h && atoi(h) > 0) hz = atoi(h);
    samples = VirtualAlloc(NULL, (SIZE_T)MAX_SAMPLES * 2 * sizeof(ULONG64), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    HMODULE exe = GetModuleHandleA(NULL);
    const IMAGE_DOS_HEADER *dos = (const IMAGE_DOS_HEADER *)exe;
    const IMAGE_NT_HEADERS *nt = (const IMAGE_NT_HEADERS *)((const char *)exe + dos->e_lfanew);
    exe_lo = (ULONG64)(ULONG_PTR)exe;
    exe_hi = exe_lo + nt->OptionalHeader.SizeOfImage;
    if (!samples) return;
    prof_thread = CreateThread(NULL, 0, prof_main, NULL, 0, NULL);
    atexit(prof_dump);
}
#else
typedef int lrprof_not_here;     // (Windows only: on Linux use perf)
#endif
