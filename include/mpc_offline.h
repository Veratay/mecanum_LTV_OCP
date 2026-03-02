#pragma once

#include "mpc_types.h"
#include <cstdio>

// Precompute all MPC windows along a reference path
// Returns dynamically allocated array of PrecomputedWindow (caller must delete[])
// n_windows_out is set to the number of windows produced
PrecomputedWindow* mpc_precompute_all(const RefNode* ref_path, int n_path,
                                      const ModelParams& params,
                                      const MPCConfig& config,
                                      int& n_windows_out);

// Serialize precomputed windows to binary file
// Returns 0 on success
int mpc_save_windows(const char* filename, const PrecomputedWindow* windows,
                     int n_windows, const MPCConfig& config);

// Load precomputed windows from binary file
// Returns dynamically allocated array (caller must delete[])
// n_windows_out and config_out are populated from the file header
PrecomputedWindow* mpc_load_windows(const char* filename, int& n_windows_out,
                                     MPCConfig& config_out);
