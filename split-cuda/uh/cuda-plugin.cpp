/****************************************************************************
 *  Copyright (C) 2019-2020 by Twinkle Jain, Rohan garg, and Gene Cooperman *
 *  jain.t@husky.neu.edu, rohgarg@ccs.neu.edu, gene@ccs.neu.edu             *
 *                                                                          *
 *  This file is part of DMTCP.                                             *
 *                                                                          *
 *  DMTCP is free software: you can redistribute it and/or                  *
 *  modify it under the terms of the GNU Lesser General Public License as   *
 *  published by the Free Software Foundation, either version 3 of the      *
 *  License, or (at your option) any later version.                         *
 *                                                                          *
 *  DMTCP is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *  GNU Lesser General Public License for more details.                     *
 *                                                                          *
 *  You should have received a copy of the GNU Lesser General Public        *
 *  License along with DMTCP:dmtcp/src.  If not, see                        *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

#ifndef _GNU_SOURCE
  #define _GNU_SOURCE
#endif
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <vector>
#include <algorithm>
#include <map>

#include "dmtcp.h"
#include "config.h"
#include "jassert.h"
#include "procmapsarea.h"
#include "util.h"
#include "log_and_replay.h"
#include "mem-wrapper.h"
#include "switch-context.h"
#include "upper-half-wrappers.h"

// #define _real_dlsym NEXT_FNC(dlsym)
// #define _real_dlopen NEXT_FNC(dlopen)
// #define _real_dlerror NEXT_FNC(dlerror)

#define DEV_NVIDIA_STR "/dev/nvidia"

using namespace dmtcp;
std::map<void *, lhckpt_pages_t>  lh_pages_maps;
void * lh_ckpt_mem_addr = NULL;
size_t lh_ckpt_mem_size = 0;
int pagesize = sysconf(_SC_PAGESIZE);
get_mmapped_list_fptr_t get_mmapped_list_fnc = NULL;
std::vector<MmapInfo_t> *uh_mmaps;
void *global_fatCubin;
void **global_fatCubinHandle;

extern "C" pid_t dmtcp_get_real_pid();
/* This function returns a range of zero or non-zero pages. If the first page
 * is non-zero, it searches for all contiguous non-zero pages and returns them.
 * If the first page is all-zero, it searches for contiguous zero pages and
 * returns them.
 */
static void
mtcp_get_next_page_range(Area *area, size_t *size, int *is_zero)
{
  char *pg;
  char *prevAddr;
  size_t count = 0;
  const size_t one_MB = (1024 * 1024);

  if (area->size < one_MB) {
    *size = area->size;
    *is_zero = 0;
    return;
  }
  *size = one_MB;
  *is_zero = Util::areZeroPages(area->addr, one_MB / MTCP_PAGE_SIZE);
  prevAddr = area->addr;
  for (pg = area->addr + one_MB;
       pg < area->addr + area->size;
       pg += one_MB) {
    size_t minsize = MIN(one_MB, (size_t)(area->addr + area->size - pg));
    if (*is_zero != Util::areZeroPages(pg, minsize / MTCP_PAGE_SIZE)) {
      break;
    }
    *size += minsize;
    if (*is_zero && ++count % 10 == 0) { // madvise every 10MB
      if (madvise(prevAddr, area->addr + *size - prevAddr,
                  MADV_DONTNEED) == -1) {
        JNOTE("error doing madvise(..., MADV_DONTNEED)")
          (JASSERT_ERRNO) ((void *)area->addr) ((int)*size);
        prevAddr = pg;
      }
    }
  }
}

static void
mtcp_write_non_rwx_and_anonymous_pages(int fd, Area *orig_area)
{
  Area area = *orig_area;

  /* Now give read permission to the anonymous/[heap]/[stack]/[stack:XXX] pages
   * that do not have read permission. We should remove the permission
   * as soon as we are done writing the area to the checkpoint image
   *
   * NOTE: Changing the permission here can results in two adjacent memory
   * areas to become one (merged), if they have similar permissions. This can
   * results in a modified /proc/self/maps file. We shouldn't get affected by
   * the changes because we are going to remove the PROT_READ later in the
   * code and that should reset the /proc/self/maps files to its original
   * condition.
   */

  JASSERT(orig_area->name[0] == '\0' || (strcmp(orig_area->name,
                                                "[heap]") == 0) ||
          (strcmp(orig_area->name, "[stack]") == 0) ||
          (Util::strStartsWith(area.name, "[stack:XXX]")));

  if ((orig_area->prot & PROT_READ) == 0) {
    JASSERT(mprotect(orig_area->addr, orig_area->size,
                     orig_area->prot | PROT_READ) == 0)
      (JASSERT_ERRNO) (orig_area->size) (orig_area->addr)
    .Text("error adding PROT_READ to mem region");
  }

  while (area.size > 0) {
    size_t size;
    int is_zero;
    Area a = area;
    if (dmtcp_infiniband_enabled && dmtcp_infiniband_enabled()) {
      size = area.size;
      is_zero = 0;
    } else {
      mtcp_get_next_page_range(&a, &size, &is_zero);
    }

    a.properties = is_zero ? DMTCP_ZERO_PAGE : 0;
    a.size = size;

    Util::writeAll(fd, &a, sizeof(a));
    if (!is_zero) {
      Util::writeAll(fd, a.addr, a.size);
    } else {
      if (madvise(a.addr, a.size, MADV_DONTNEED) == -1) {
        JNOTE("error doing madvise(..., MADV_DONTNEED)")
          (JASSERT_ERRNO) (a.addr) ((int)a.size);
      }
    }
    area.addr += size;
    area.size -= size;
  }

  /* Now remove the PROT_READ from the area if it didn't have it originally
  */
  if ((orig_area->prot & PROT_READ) == 0) {
    JASSERT(mprotect(orig_area->addr, orig_area->size, orig_area->prot) == 0)
      (JASSERT_ERRNO) (orig_area->addr) (orig_area->size)
    .Text("error removing PROT_READ from mem region.");
  }
}

// Returns true if needle is in the haystack
static inline int
regionContains(const void *haystackStart,
               const void *haystackEnd,
               const void *needleStart,
               const void *needleEnd)
{
  return needleStart >= haystackStart && needleEnd <= haystackEnd;
}

/*
  This function checks whether we should skip the region or checkpoint fully or
  Partially.
  The idea is that we are recording each mmap by upper-half. So, all the
  ckpt'ble area
*/

#undef dmtcp_skip_memory_region_ckpting
EXTERNC int
dmtcp_skip_memory_region_ckpting(ProcMapsArea *area)
{
  JNOTE("In skip area");
  ssize_t rc = 1;
  if (strstr(area->name, "vvar") ||
    strstr(area->name, "vdso") ||
    strstr(area->name, "vsyscall") ||
    strstr(area->name, DEV_NVIDIA_STR)) {
    return rc; // skip this region
  }

  if (strstr(area->name, "heap")) {
    JTRACE("Ignoring heap region")(area->name)((void*)area->addr);
    return 1;
  }

  // If it's the upper-half stack, don't skip
  if (area->addr >= lh_info->uh_stack_start && area->endAddr <= lh_info->uh_stack_end) {
    return 0;
  }

  get_mmapped_list_fnc = (get_mmapped_list_fptr_t) lh_info->mmap_list_fptr;

  int numUhRegions;
  if (uh_mmaps == NULL) {
    uh_mmaps = get_mmapped_list_fnc(&numUhRegions);
  }

  for (MmapInfo_t &region : *uh_mmaps) {
    if (regionContains(region.addr, region.addr + region.len,
                       area->addr, area->endAddr)) {
      return 0;
    }
    if (regionContains(area->addr, area->endAddr,
                       region.addr, region.addr + region.len)) {
      area->addr = (char*) region.addr;
      area->endAddr = (char*) (region.addr + region.len);
      area->size = area->endAddr - area->addr;
      return 0;
    }
    if (area->addr < region.addr && region.addr < area->endAddr &&
        area->endAddr < region.addr + region.len) {
      area->addr = (char*) region.addr;
      area->size = area->endAddr - area->addr;
      return 0;
    }
    if (region.addr < area->addr && area->addr < region.addr + region.len &&
        region.addr + region.len < area->endAddr) {
      area->endAddr = (char*) (region.addr + region.len);
      area->size = area->endAddr - area->addr;
      return 0;
    }
  }
  return 1;
}

void save_lh_pages_to_memory()
{
  // get the Lower-half page maps
  lh_pages_maps = getLhPageMaps();

  size_t total_size = sizeof(int);
  for (auto lh_page : lh_pages_maps) {
    // printf("\n Address = %p with size = %lu", lh_page.first,
    // lh_page.second.mem_len);
    // lhckpt_pages_t
    total_size += lh_page.second.mem_len + sizeof(lh_page.second);
  }
  if (total_size > 0) {
    // round up to the page size
    total_size = ((total_size + pagesize - 1) & ~(pagesize - 1));
    // mmap a region in the process address space big enough for the structure
    // + data + initial_guard_page
    void *addr = mmap(NULL, pagesize + total_size + pagesize,
                      PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    JASSERT(addr != MAP_FAILED) (addr) (JASSERT_ERRNO);
    JASSERT(mprotect(addr, pagesize, PROT_EXEC) != -1)(addr)(JASSERT_ERRNO);
    addr = (void *)((VA)addr + pagesize);
    JASSERT(mprotect((void *)((VA)addr+total_size), pagesize, PROT_EXEC) != -1)\
      (addr)(JASSERT_ERRNO)(total_size)(pagesize);

    // make this address and size available to dmtcp_skip_memory_region
    lh_ckpt_mem_addr = addr;
    lh_ckpt_mem_size = total_size;

    size_t count = 0;
    int total_entries = lh_pages_maps.size();
    memcpy(((VA)addr + count), &total_entries, sizeof total_entries);
    count += sizeof(total_entries);
    // mprotect with read permission on the page
    /* Que: should we change the read permission of the entire cuda malloc'ed
      region at once? So that when we mprotect back to the ---p permission, we
      don't see many entries in /proc/pid/maps file even if the perms are same.
    */
    for (auto lh_page : lh_pages_maps) {
      void *mem_addr = lh_page.second.mem_addr;
      size_t mem_len = lh_page.second.mem_len;
      // copy the metadata and data to the new mmap'ed region
      void * dest = memcpy(((VA)addr + count), (void *)&lh_page.second,
                  sizeof(lh_page.second));
      JASSERT(dest == ((VA)addr + count))("memcpy failed") (addr)
              (dest) (count) (sizeof(lh_page.second)) (JASSERT_ERRNO);
      count += sizeof(lh_page.second);
      // copy the actual data
      switch (lh_page.second.mem_type) {
        case (CUDA_MALLOC_PAGE):
        case (CUDA_UVM_PAGE):
        {
          cudaMemcpy(((VA)addr + count), mem_addr, mem_len, \
                     cudaMemcpyDeviceToHost);
          break;
        }
        default:
        {
          JASSERT(false) ("page type unkown");
        }
      }
      // JASSERT(dest == (void *)((uint64_t)addr + count))("memcpy failed")
      //  (addr) (count) (mem_addr) (mem_len);
      count += mem_len;
    }
  }
}

void restore_device_pages() {
  lh_pages_maps = getLhPageMaps();

  // read total entries count
  int total_entries = 0;
  int count = 0;
  memcpy(&total_entries, ((VA)lh_ckpt_mem_addr+count), sizeof (total_entries));
  count += sizeof (total_entries);
  for (int i = 0; i < total_entries; i++) {
    // read metadata of one entry
    lhckpt_pages_t lhpage_info;
    memcpy(&lhpage_info, ((VA)lh_ckpt_mem_addr+count), sizeof (lhpage_info));
    count += sizeof(lhpage_info);

    void *dest_addr = lhpage_info.mem_addr;
    size_t size = lhpage_info.mem_len;

    switch (lhpage_info.mem_type) {
      case (CUDA_UVM_PAGE):
      case (CUDA_MALLOC_PAGE):
      {
        // copy back the actual data
        cudaMemcpy(dest_addr, ((VA)lh_ckpt_mem_addr+count), size, cudaMemcpyHostToDevice);
        count += size;
        break;
      }
      default:
        printf("page type not implemented\n");
        break;
    }
  }
}

void pre_ckpt()
{
  /**/
  disableLogging();
  cudaDeviceSynchronize();
  save_lh_pages_to_memory();
  enableLogging();
}

void resume()
{
  // unmap the region we mapped it earlier
  if (lh_ckpt_mem_addr != NULL && lh_ckpt_mem_size > 0)
  {
     JASSERT(munmap(lh_ckpt_mem_addr, lh_ckpt_mem_size) != -1)
            ("munmap failed!") (lh_ckpt_mem_addr) (lh_ckpt_mem_size);
     JASSERT(munmap((VA)lh_ckpt_mem_addr - pagesize, pagesize) != -1)
            ("munmap failed!") ((VA)lh_ckpt_mem_addr - pagesize) (pagesize);
  } else {
    JTRACE("no memory region was allocated earlier")
          (lh_ckpt_mem_addr) (lh_ckpt_mem_size);
  }
}

void restart()
{
  reset_wrappers();
  initialize_wrappers();
  disableLogging();
  logs_read_and_apply();
  restore_device_pages();
  enableLogging();
}

static void
cuda_plugin_event_hook(DmtcpEvent_t event, DmtcpEventData_t *data)
{
  switch (event) {
    case DMTCP_EVENT_INIT:
    {
      JTRACE("*** DMTCP_EVENT_INIT");
      JTRACE("Plugin intialized");
      break;
    }
    case DMTCP_EVENT_EXIT:
    {
      JTRACE("*** DMTCP_EVENT_EXIT");
      break;
    }
    case DMTCP_EVENT_PRECHECKPOINT:
    {
      pre_ckpt();
      break;
    }
    case DMTCP_EVENT_RESUME:
    {
      resume();
      break;
    }
    case DMTCP_EVENT_RESTART:
    {
      restart();
      break;
    }
    default:
      break;
  }
}

/*
static DmtcpBarrier cudaPluginBarriers[] = {
  { DMTCP_GLOBAL_BARRIER_PRE_CKPT, pre_ckpt, "checkpoint" },
  { DMTCP_GLOBAL_BARRIER_RESUME, resume, "resume" },
  { DMTCP_GLOBAL_BARRIER_RESTART, restart, "restart" }
};
*/
DmtcpPluginDescriptor_t cuda_plugin = {
  DMTCP_PLUGIN_API_VERSION,
  PACKAGE_VERSION,
  "cuda_plugin",
  "DMTCP",
  "dmtcp@ccs.neu.edu",
  "Cuda Split Plugin",
  cuda_plugin_event_hook
};
//  DMTCP_DECL_BARRIERS(cudaPluginBarriers),

DMTCP_DECL_PLUGIN(cuda_plugin);
