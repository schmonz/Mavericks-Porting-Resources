// Replacement for osu!'s Velopack UpdateMac binary on Mac OS X 10.9.
// Applying stock updates would both crash (the real UpdateMac needs
// 10.10+ APIs) and overwrite the Mavericks compatibility patches, so
// this stub satisfies Velopack's process contract without installing
// anything: it outlives the "exited too soon" check, waits for the
// game to exit, deletes any staged .nupkg packages, and relaunches
// the game when Velopack asked for a restart.
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <dirent.h>
#include <time.h>
#include <libgen.h>
#include <mach-o/dyld.h>

static FILE *logf;

static void logmsg(const char *fmt, ...)
{
    if (!logf)
        return;
    va_list ap;
    time_t now = time(NULL);
    char ts[64];
    strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", localtime(&now));
    fprintf(logf, "%s ", ts);
    va_start(ap, fmt);
    vfprintf(logf, fmt, ap);
    va_end(ap);
    fprintf(logf, "\n");
    fflush(logf);
}

int main(int argc, char **argv)
{
    char logpath[1024];
    snprintf(logpath, sizeof logpath, "%s/Library/Logs/osu-updatemac-stub.log", getenv("HOME") ?: "/tmp");
    logf = fopen(logpath, "a");

    char cmdline[2048] = "";
    for (int i = 1; i < argc; i++) {
        strlcat(cmdline, argv[i], sizeof cmdline);
        strlcat(cmdline, " ", sizeof cmdline);
    }
    logmsg("invoked: %s", cmdline);

    long wait_pid = 0;
    const char *package_dir = NULL;
    int restart = 1;
    int apply = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--waitPid") && i + 1 < argc)
            wait_pid = strtol(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--packageDir") && i + 1 < argc)
            package_dir = argv[++i];
        else if (!strcmp(argv[i], "--norestart"))
            restart = 0;
        else if (!strcmp(argv[i], "apply"))
            apply = 1;
        else if (!strcmp(argv[i], "start"))
            apply = 0;
    }

    // Outlive the caller's "exited too soon" check.
    sleep(3);

    if (wait_pid > 0) {
        logmsg("waiting for pid %ld to exit", wait_pid);
        for (int i = 0; i < 600; i++) {
            if (kill((pid_t)wait_pid, 0) != 0 && errno == ESRCH)
                break;
            usleep(100000);
        }
    }

    if (apply && package_dir) {
        DIR *d = opendir(package_dir);
        if (d) {
            struct dirent *e;
            while ((e = readdir(d)) != NULL) {
                size_t len = strlen(e->d_name);
                if (len > 6 && !strcmp(e->d_name + len - 6, ".nupkg")) {
                    char path[2048];
                    snprintf(path, sizeof path, "%s/%s", package_dir, e->d_name);
                    if (unlink(path) == 0)
                        logmsg("discarded staged package: %s", e->d_name);
                    else
                        logmsg("could not delete %s: %s", path, strerror(errno));
                }
            }
            closedir(d);
        }
    }

    if (restart) {
        // Relaunch the game via the wrapper script next to this binary.
        char self[1024];
        uint32_t size = sizeof self;
        if (_NSGetExecutablePath(self, &size) == 0) {
            char *dir = dirname(self);
            char launcher[1088];
            snprintf(launcher, sizeof launcher, "%s/osu", dir);
            logmsg("relaunching %s", launcher);
            pid_t pid = fork();
            if (pid == 0) {
                setsid();
                execl(launcher, launcher, (char *)NULL);
                _exit(127);
            }
        }
    }

    logmsg("done");
    return 0;
}
