// view.c - real-time viewer: fly through the scene with both backends of the engine.
//
//   make && bin/view [options]           (run with -h for the list)
//
// The viewer itself is part of the engine (app/viewer.h): this is its options and its loop.
#include "app/viewer.h"

int main(int argc, char **argv){
    options_parse(viewer_options(), PROG_VIEW, argc, argv);
    if (viewer_setup_scene() != 0 || viewer_open("view") != 0) return 1;
    viewer_print_controls();
    viewer_run();
    return 0;
}
