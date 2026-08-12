/*
 * dnssd_process_result.c -- MavericksLegacySupport libSystem shim.
 * Shares the DBG helper and common includes with the other shims via
 * mav_shim_debug.h.
 */
#include "mav_shim_debug.h"

/* ── DNSServiceProcessResult shim: the call reads one reply message off the
 * mDNSResponder client socket with a *blocking* read_all(). It is meant to be
 * called only when the socket from DNSServiceRefSockFD() is readable, and it
 * assumes a whole message is waiting. Neither assumption is guaranteed here.
 *
 * Bun drives that socket from its event loop, which reaches the kernel through
 * kevent64 -- a call 10.9 lacks, so it arrives via the emulation in
 * kevent64_shim.c. Any readiness that the emulation reports without a message
 * actually being available (and any second report of a readiness the caller
 * already consumed) turns this call into an unbounded recvfrom on the main
 * thread: the event loop stops, and with it every timer, socket and pending
 * request in the process. The user sees Claude Code start and then hang with
 * no output and no error.
 *
 * Poll the socket first and return "nothing happened" when no message is
 * waiting, which is what the caller's own contract already allows. When a
 * message is waiting, bound the read so a message that is somehow split across
 * writes surfaces as a DNS error the caller can retry rather than as a
 * permanently parked process. */
#include <dns_sd.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>

/* Long enough that it never trips on the tail of a real message, short enough
 * that a truncated one fails visibly instead of hanging. */
#define DNSSD_TAIL_TIMEOUT_SEC 5

DNSServiceErrorType DNSServiceProcessResult_wrapper(DNSServiceRef sdRef)
    __asm("_DNSServiceProcessResult");
DNSServiceErrorType DNSServiceProcessResult_wrapper(DNSServiceRef sdRef) {
    static DNSServiceErrorType (*real_process)(DNSServiceRef) = NULL;
    static int (*real_sockfd)(DNSServiceRef) = NULL;
    if (!real_process) real_process = dlsym(RTLD_NEXT, "DNSServiceProcessResult");
    if (!real_sockfd)  real_sockfd  = dlsym(RTLD_NEXT, "DNSServiceRefSockFD");
    if (!real_process) return kDNSServiceErr_Unknown;

    int fd = real_sockfd ? real_sockfd(sdRef) : -1;
    if (fd < 0) return real_process(sdRef);

    struct pollfd pfd = { fd, POLLIN, 0 };
    int ready = poll(&pfd, 1, 0);
    if (ready == 0 || (ready > 0 && !(pfd.revents & (POLLIN | POLLHUP | POLLERR)))) {
        DBG("DNSServiceProcessResult fd=%d: no message waiting, skipping read", fd);
        return kDNSServiceErr_NoError;
    }

    struct timeval tv = { DNSSD_TAIL_TIMEOUT_SEC, 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    DNSServiceErrorType err = real_process(sdRef);
    if (err) DBG("DNSServiceProcessResult fd=%d -> err %d", fd, (int)err);
    return err;
}
