#!/usr/bin/env python3
"""Checks engine/gpu/vk/vk.h (our own declarations of the Vulkan API) against the official
headers: every struct's size and every field's offset, and every constant we define that the
official headers also have. A development check, run where the Vulkan headers are installed
(MSYS2: mingw-w64-ucrt-x86_64-vulkan-headers):

    python tools/vk_check.py            (uses cc on the PATH; CC=... to choose)
"""
import os, re, subprocess, sys, tempfile

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
src = open(os.path.join(root, 'engine', 'gpu', 'vk', 'vk.h')).read()
body = re.sub(r'//[^\n]*', '', src)

structs = []   # (name, [fields])
for m in re.finditer(r'typedef struct (\w+)\s*\{(.*?)\}\s*(\w+);', body, re.S):
    name, fields = m.group(3), []
    for decl in m.group(2).split(';'):
        decl = decl.strip()
        if not decl: continue
        decl = re.sub(r'\s+', ' ', decl)
        # "type a, b[3], *c" -> names
        first = re.match(r'((?:const )?[\w ]*?[\w*]+[ *]+)(.*)', decl)
        names = decl.split(',')
        head = names[0]
        nm = re.findall(r'(\w+)\s*(?:\[[^\]]*\])?\s*$', head)
        out = [nm[0]] if nm else []
        for n in names[1:]:
            k = re.findall(r'(\w+)\s*(?:\[[^\]]*\])?\s*$', n.strip())
            if k: out.append(k[0])
        fields += out
    structs.append((name, fields))

consts = re.findall(r'^\s*(VK_[A-Z0-9_]+)\s*=\s*', body, re.M) + \
         re.findall(r'^#define (VK_[A-Z0-9_]+) [^\n(]', body, re.M)
skip = {'VK_HANDLE', 'VK_HANDLE64', 'VK_MAKE_API_VERSION', 'VK_API_VERSION_1_3'}
consts = [c for c in dict.fromkeys(consts) if c not in skip and not c.endswith('MAX_ENUM')]

def program(header, prefix):
    lines = ['#include <stdio.h>', '#include <stddef.h>', header, 'int main(void){']
    for name, fields in structs:
        lines.append(f'printf("{name} %u\\n", (unsigned)sizeof({name}));')
        for f in fields:
            lines.append(f'printf("{name}.{f} %u\\n", (unsigned)offsetof({name}, {f}));')
    for c in consts:
        lines.append(f'printf("{c} %lld\\n", (long long)({c}));')
    lines.append('return 0; }')
    return '\n'.join(lines)

cc = os.environ.get('CC', 'cc')
outs = []
with tempfile.TemporaryDirectory() as d:
    for tag, header in (('ours', f'#include "{os.path.join(root, "engine", "gpu", "vk", "vk.h")}"'),
                        ('ref', '#define VK_USE_PLATFORM_WIN32_KHR\n#include <vulkan/vulkan.h>' if os.name == 'nt'
                                else '#include <vulkan/vulkan.h>')):
        c = os.path.join(d, tag + '.c'); exe = os.path.join(d, tag + '.exe')
        open(c, 'w').write(program(header, tag))
        r = subprocess.run([cc, c, '-o', exe], capture_output=True, text=True)
        if r.returncode:
            print(f'{tag}: does not compile\n{r.stderr[:3000]}'); sys.exit(1)
        outs.append(subprocess.run([exe], capture_output=True, text=True).stdout.splitlines())
bad = [(a, b) for a, b in zip(*outs) if a != b]
for a, b in bad: print(f'differs: ours {a}  official {b}')
print(f'{len(outs[0])} sizes, offsets and constants checked, {len(bad)} differ')
sys.exit(1 if bad else 0)
