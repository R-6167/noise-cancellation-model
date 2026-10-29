/* Minimal C host using noise_cancellation (link libnoise_cancellation). */
#include "noise_cancellation.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    NcHandle *h = NULL;
    NcStatus st = nc_create(NC_ENGINE_RNNOISE, NULL, 0.85f, &h);
    if (st != NC_OK) {
        fprintf(stderr, "create failed: %s\n", nc_status_string(st));
        return 1;
    }

    float in[NC_FRAME_SIZE] = {0};
    float out[NC_FRAME_SIZE] = {0};
    st = nc_process(h, in, out, NC_FRAME_SIZE);
    if (st != NC_OK) {
        fprintf(stderr, "process failed: %s\n", nc_status_string(st));
    }

    const char *name = NULL;
    nc_engine_name(h, &name);
    printf("engine=%s sample_rate=%u frame=%zu\n",
           name ? name : "?", nc_sample_rate(), nc_frame_size());

    nc_destroy(h);
    return 0;
}
