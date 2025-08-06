#include <assert.h>
#include <errno.h>
#include <sys/mman.h>
#include <unistd.h>


#ifdef STANDALONE
// Returns on success; asserts on error.
int main() {
  long pagesize = sysconf(_SC_PAGESIZE); 
  errno = 0;
  int test_flag = MAP_PRIVATE | MAP_ANONYMOUS;
  void *addr = mmap(NULL, pagesize, 0, test_flag, -1, 0); // No clobber memory
  assert(addr != MAP_FAILED);

  void *addr2 = mmap_fixed_noreplace(addr, pagesize, 0, test_flag, -1, 0);
  assert(addr2 == MAP_FAILED && errno == EEXIST);

  addr2 = mmap(addr, pagesize, 0, test_flag, -1, 0); // Test internals of fnc
  assert(addr2 != MAP_FAILED && addr2 != addr); // memory region already exists
  assert(munmap(addr, pagesize) == 0);
  assert(munmap(addr2, pagesize) == 0);

  addr2 = mmap_fixed_noreplace(addr, pagesize, 0, test_flag, -1, 0);
  assert(addr2 != MAP_FAILED && addr2 == addr);
  assert(munmap(addr, pagesize) == 0);

#if MAP_FIXED_NOREPLACE != -1
  addr2 = mmap_fixed_noreplace(addr, pagesize, 0,
                               test_flag | MAP_FIXED_NOREPLACE, -1, 0);
  assert(addr2 != MAP_FAILED && addr2 == addr);
  addr2 = mmap_fixed_noreplace(addr, pagesize, 0,
                               test_flag | MAP_FIXED_NOREPLACE, -1, 0);
  assert(addr2 == MAP_FAILED && errno == EEXIST);
  assert(munmap(addr, pagesize) == 0);
#endif

  return 0;
}
#endif
