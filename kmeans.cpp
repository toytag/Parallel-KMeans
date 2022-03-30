////////////////
// 
// File: kmeans.cpp
//
//  Main body of K-Means simulaton. Reads in the original data points from
//  `ori.txt`, performs K-Means clustering on randomly-picked initial
//  centers, and writes the results into `res.txt` with the same format.
//
//  * You may (and should) include some extra headers for optimizations.
//
//  * You should and ONLY should modify the function body of `kmeans()`.
//    DO NOT change any other exitsing part of the program.
//
//  * You may add your own auxiliary functions if you wish. Extra declarations
//    can go in `kmeans.h`.
//
// Jose @ ShanghaiTech University
//
////////////////

#include <fstream>
#include <limits>
#include <math.h>
#include <chrono>
#include "kmeans.h"


/*********************************************************
        Your extra headers and static declarations
 *********************************************************/
#include <omp.h>
#include <vector>
#include <immintrin.h>
/*********************************************************
                           End
 *********************************************************/


/*
 * Entrance point. Time ticking will be performed, so it will be better if
 *   you have cleared the cache for precise profiling.
 *
 */
int
main (int argc, char *argv[])
{
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input.txt> <output.txt>"
                  << std::endl;
        return -1;
    }
    if (!(bool)std::ifstream(argv[1])) {
        std::cerr << "ERROR: Data file " << argv[1] << " does not exist!"
                  << std::endl;
        return -1;
    }
    if ((bool)std::ifstream(argv[2])) {
        std::cerr << "ERROR: Destination " << argv[2] << " already exists!"
                  << std::endl;
        return -1;
    }
    FILE *fi = fopen(argv[1], "r"), *fo = fopen(argv[2], "w");
    
    /* From `ori.txt`, acquire dataset size, number of colors (i.e. K in
       K-Means),and read in all data points into static array `data`. */
    int pn, cn;

    if (fscanf(fi, "%d / %d\n", &pn, &cn)) {};

    point_t * const data = new point_t[pn];
    color_t * const coloring = new color_t[pn];

    for (int i = 0; i < pn; ++i)
        coloring[i] = 0;

    int i = 0, c;
    double x, y;

    while (fscanf(fi, "%lf, %lf, %d\n", &x, &y, &c) == 3) {
        data[i++].setXY(x, y);
        if (c < 0 || c >= cn) {
            std::cerr << "ERROR: Invalid color code encoutered!"
                      << std::endl;
            return -1;
        }
    }
    if (i != pn) {
        std::cerr << "ERROR: Number of data points inconsistent!"
                  << std::endl;
        return -1;
    }

    /* Generate a random set of initial center points. */
    point_t * const mean = new point_t[cn];

    srand(5201314);
    for (int i = 0; i < cn; ++i) {
        int idx = rand() % pn;
        mean[i].setXY(data[idx].getX(), data[idx].getY());
    }

    /* Invode K-Means algorithm on the original dataset. It should cluster
       the data points in `data` and assign their color codes to the
       corresponding entry in `coloring`, using `mean` to store the center
       points. */
    std::cout << "Doing K-Means clustering on " << pn
              << " points with K = " << cn << "..." << std::flush;
    auto ts = std::chrono::high_resolution_clock::now();
    kmeans(data, mean, coloring, pn, cn);
    auto te = std::chrono::high_resolution_clock::now();
    std::cout << "done." << std::endl;
    std::cout << " Total time elapsed: "
              << std::chrono::duration_cast<std::chrono::milliseconds> \
                 (te - ts).count()
              << " milliseconds." << std::endl; 

    /* Write the final results to `res.txt`, in the same format as input. */
    fprintf(fo, "%d / %d\n", pn, cn);
    for (i = 0; i < pn; ++i)
        fprintf(fo, "%.8lf, %.8lf, %d\n", data[i].getX(), data[i].getY(),
                coloring[i]);

    /* Free the resources and return. */
    delete[](data);
    delete[](coloring);
    delete[](mean);
    fclose(fi);
    fclose(fo);
    return 0;
}


/*********************************************************
           Feel free to modify the things below
 *********************************************************/

/*
 * K-Means algorithm clustering. Originally implemented in a traditional
 *   sequential way. You should optimize and parallelize it for a better
 *   performance. Techniques you can use include but not limited to:
 *
 *     1. OpenMP shared-memory parallelization.
 *     2. SSE SIMD instructions.
 *     3. Cache optimizations.
 *     4. Manually using pthread.
 *     5. ...
 *
 */
void
kmeans (point_t * const data, point_t * const mean, color_t * const coloring,
        const int pn, const int cn)
{
    bool converge = true;
    int num_threads = omp_get_max_threads();
    int block_size = pn / num_threads;
    /* ------------------ HYPERPARAMETERS ------------------ */

    /* Loop through the following two stages until no point changes its color
       during an iteration. */
    do {
        converge = true;

        std::vector<int> count(cn, 0);
        std::vector<double> sum_x(cn, 0), sum_y(cn, 0);

        /* Compute the color of each point. A point gets assigned to the
           cluster with the nearest center point. */
        omp_set_num_threads(num_threads);
        #pragma omp parallel
        {

            std::vector<int> local_count(cn, 0);
            std::vector<double> local_sum_x(cn, 0), local_sum_y(cn, 0);

            int id = omp_get_thread_num();
            int end = ((id + 1) == num_threads) ? pn : ((id + 1) * block_size);
            for (int i = id * block_size; i < end; i += 4) {
                color_t new_color[4] = {0, 0, 0, 0};

                __m256d min_dist = _mm256_set1_pd(std::numeric_limits<double>::infinity());
                __m256d vec8_x = _mm256_setr_pd(data[i].x, data[i+1].x, data[i+2].x, data[i+3].x);
                __m256d vec8_y = _mm256_setr_pd(data[i].y, data[i+1].y, data[i+2].y, data[i+3].y);

                for (color_t c = 0; c < cn; c++) {

                    __m256d dist_x = _mm256_sub_pd(vec8_x, _mm256_set1_pd(mean[c].x));
                    dist_x = _mm256_mul_pd(dist_x, dist_x);
                    __m256d dist_y = _mm256_sub_pd(vec8_y, _mm256_set1_pd(mean[c].y));
                    dist_y = _mm256_mul_pd(dist_y, dist_y);
                    __m256d dist = _mm256_add_pd(dist_x, dist_y);
                    __m256d new_min_dist = _mm256_min_pd(min_dist, dist);

                    for (int j = 0; j < 4; j++) {
                        int64_t md = _mm256_extract_epi64(_mm256_castpd_si256(min_dist), j);
                        int64_t nmd = _mm256_extract_epi64(_mm256_castpd_si256(new_min_dist), j);
                        if (md != nmd) {
                            new_color[j] = c;
                        }
                    }

                    min_dist = new_min_dist;

                }

                for (int j = 0; j < 4; j++) {
                    if (coloring[i+j] != new_color[j]) {
                        coloring[i+j] = new_color[j];
                        converge = false;
                    }
                    local_sum_x[new_color[j]] += data[i+j].x;
                    local_sum_y[new_color[j]] += data[i+j].y;
                    local_count[new_color[j]] += 1;
                }

            }

            // not divisible
            for (int i = end/4*4; i < end; i++) {
                color_t new_color = cn;
                double min_dist = std::numeric_limits<double>::infinity();

                for (color_t c = 0; c < cn; c++) {
                    double dist = pow(data[i].x - mean[c].x, 2) + pow(data[i].y - mean[c].y, 2);
                    if (dist < min_dist) {
                        min_dist = dist;
                        new_color = c;
                    }
                }

                if (coloring[i] != new_color) {
                    coloring[i] = new_color;
                    converge = false;
                }

                local_sum_x[new_color] += data[i].x;
                local_sum_y[new_color] += data[i].y;
                local_count[new_color] += 1;
            }

            #pragma omp critical
            {
                for (color_t c = 0; c < cn; c++) {
                    sum_x[c] += local_sum_x[c];
                    sum_y[c] += local_sum_y[c];
                    count[c] += local_count[c];
                }
            }

        }

        /* Calculate the new mean for each cluster to be the current average
           of point positions in the cluster. */
        omp_set_num_threads(2);
        #pragma omp parallel for schedule(static)
        for (color_t c = 0; c < cn; ++c)
            mean[c].setXY(sum_x[c] / count[c], sum_y[c] / count[c]);
        
    } while (!converge);
}

/*********************************************************
                           End
 *********************************************************/
