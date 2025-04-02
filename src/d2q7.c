#define _XOPEN_SOURCE 600
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>

#define SILENT 1

#define DIRECTIONS 7
#define ALPHA 0.5
#define TAU 1.0

typedef float real_t;
#define MPI_REAL_T MPI_FLOAT

typedef enum {
    SOLID,
    WALL,
    FLUID
} domain_t;

int OFFSETS[2][DIRECTIONS][2] = {
    { {0,1}, {1,1}, { 1,0}, {0,-1}, {-1, 0}, {-1,1}, {0,0} },  /* Odd rows */
    { {0,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1}, {-1,0}, {0,0} } /* Even rows */
};

void init_domain(void);     // Initialize domain geometry from input file
void init_mpi_cart_grid(void);  // Initialize MPI Cartesian grid
void init_mpi_types(void);      // Initialize MPI Custom datatypes
void scatter_domain(void);
void collide(void);         // Collision step
void border_exchange(void); // MPI border exchange
void stream(void);          // Streaming step

void save(int iteration);       // Store results
void options(int argc, char **argv); // Command line arguments

int W, H;      // Width and height of domain
int timesteps; // Number of timesteps in simulation
int store_freq; // Frequency of

domain_t *lattice = NULL; // Domain geometry
real_t *densities[2] = {
    NULL,                 // Densities in current timestep
    NULL                  // Densities in next timestep
};
real_t *v = NULL;         // Velocities
real_t e[DIRECTIONS][2];           // Directinal vectors

real_t force[2] = {
    0.00, // External force in y direction
    0.007,  // External force in x direction
};

float *outbuf = NULL; // Output buffer (Note that this is a float)

#define LATTICE(i,j) lattice[(i)*(local_W+2)+(j)]

#define D_now(i,j,d) densities[0][(d)*(local_W+2)*(local_H+2)+(i)*(local_W+2)+(j)]
#define D_nxt(i,j,d) densities[1][(d)*(local_W+2)*(local_H+2)+(i)*(local_W+2)+(j)]

#define V_y(i,j) v[2*((i)*(local_W+2)+(j))]
#define V_x(i,j) v[2*((i)*(local_W+2)+(j))+1]

#define OUTBUF(i,j) outbuf[(i)*(local_W)+(j)]


/* MPI */
int rank; // MPI rank
int comm_size; // Total number of ranks

typedef enum { NORTH, EAST, SOUTH, WEST } Direction;
MPI_Comm comm_cart; // Cartesian communicator
int dims[2];        // Dimensions of cartesian grid [y, x]
int cart_pos[2];    // Position in cartesian grid   [y, x]
int cart_nbo[4];    // Neighbors in grid            [N, E, S, W]
MPI_Datatype subgrid;       // Datatype for local grid in global grid
MPI_Datatype column, row;   // Column and row in subgrid (including halo)
MPI_Datatype columns, rows;

int local_H, local_W;

#define MPI_RANK_ROOT 0

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (rank == MPI_RANK_ROOT) {
        options(argc, argv);
    }

    MPI_Bcast(&W, 1, MPI_INT, MPI_RANK_ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&H, 1, MPI_INT, MPI_RANK_ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&timesteps, 1, MPI_INT, MPI_RANK_ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&store_freq, 1, MPI_INT, MPI_RANK_ROOT, MPI_COMM_WORLD);

    init_mpi_cart_grid();
    init_mpi_types();

    lattice = malloc((local_W+2) * (local_H+2) * sizeof(domain_t));
    densities[0] = malloc(7 * (local_W+2) * (local_H+2) * sizeof(real_t));
    densities[1] = malloc(7 * (local_W+2) * (local_H+2) * sizeof(real_t));
    v = malloc(2 * (local_H+2) * (local_W+2) * sizeof(real_t));
    outbuf = malloc((local_H) * (local_W) * sizeof(float));

    init_domain();

    for (int i = 0; i < local_H+2; i++) {
        for (int j = 0; j < local_W+2; j++) {
            for (int d = 0; d < DIRECTIONS; d++) {
                D_nxt(i,j,d) = D_now(i,j,d) = 1.0 / 7.0;
            }
        }
    }

    for(int d=0; d<6; d++) {
        e[d][0] = sin(M_PI * d / 3.0); // y
        e[d][1] = cos(M_PI * d / 3.0); // x
    }
    e[6][0] = 0.0;
    e[6][1] = 0.0;

    double start_time, end_time, begin_time, collide_time, exchange_time, stream_time;
    double collide_total = 0, exchange_total = 0, stream_total = 0;

    start_time = MPI_Wtime();

    for (int i = 0; i < timesteps; i++) {
        begin_time = MPI_Wtime();
        collide();
        collide_time = MPI_Wtime();
        border_exchange();
        exchange_time = MPI_Wtime();
        stream();
        stream_time = MPI_Wtime();

        collide_total += collide_time - begin_time;
        exchange_total += exchange_time - collide_time;
        stream_total += stream_time - exchange_time;


        if (!SILENT && i % store_freq == 0) {
            if (rank == MPI_RANK_ROOT)
                printf("Iteration %d/%d\n", i, timesteps);
            save(i/store_freq);
        }
    }

    end_time = MPI_Wtime();

    double r_start_time, r_end_time, r_collide_total, r_exchange_total, r_stream_total;
    MPI_Reduce(&start_time, &r_start_time, 1, MPI_DOUBLE, MPI_SUM, MPI_RANK_ROOT, comm_cart);
    MPI_Reduce(&end_time, &r_end_time, 1, MPI_DOUBLE, MPI_SUM, MPI_RANK_ROOT, comm_cart);
    MPI_Reduce(&collide_total, &r_collide_total, 1, MPI_DOUBLE, MPI_SUM, MPI_RANK_ROOT, comm_cart);
    MPI_Reduce(&exchange_total, &r_exchange_total, 1, MPI_DOUBLE, MPI_SUM, MPI_RANK_ROOT, comm_cart);
    MPI_Reduce(&stream_total, &r_stream_total, 1, MPI_DOUBLE, MPI_SUM, MPI_RANK_ROOT, comm_cart);

    if (rank == MPI_RANK_ROOT) {
        /*
        printf("==== Results ====\n");
        printf("Height            %d\n", H);
        printf("Width             %d\n", W);
        printf("MPI ranks         %d\n", comm_size);
        printf("OMP threads       %d\n", omp_get_max_threads());
        printf("Iterations        %d\n", timesteps);
        printf("Elapsed time (s)  %lf\n", (r_end_time - r_start_time)/comm_size);
        printf("    Collision     %lf\n", r_collide_total/comm_size);
        printf("    Exchange      %lf\n", r_exchange_total/comm_size);
        printf("    Streaming     %lf\n", r_stream_total/comm_size);
        */

        printf("%.4lf %.4lf %.4lf %.4lf\n", (r_end_time - r_start_time)/comm_size,
            r_collide_total/comm_size, r_exchange_total/comm_size, r_stream_total/comm_size);
    }

    save(timesteps/store_freq);

    MPI_Type_free(&column);
    MPI_Type_free(&row);
    MPI_Type_free(&subgrid);

    free(lattice);
    free(densities[0]);
    free(densities[1]);
    free(v);
    free(outbuf);

    MPI_Finalize();

    return EXIT_SUCCESS;
}

void init_mpi_cart_grid(void)
{
    int periods[2] = { 1, 1 };

    MPI_Dims_create(comm_size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_cart);
    MPI_Cart_coords(comm_cart, rank, 2, cart_pos);
    MPI_Cart_shift(comm_cart, 0, 1, &cart_nbo[NORTH], &cart_nbo[SOUTH]);
    MPI_Cart_shift(comm_cart, 1, 1, &cart_nbo[WEST], &cart_nbo[EAST]);

    local_H = H / dims[0];
    local_W = W / dims[1];
}

void init_mpi_types(void)
{
    int start[2] = { cart_pos[0]*local_H, cart_pos[1]*local_W };
    int subgrid_size[2] = { local_H, local_W };
    int grid_size[2] = { H, W };

    MPI_Type_create_subarray(2, grid_size, subgrid_size, start, MPI_ORDER_C, MPI_REAL_T, &subgrid);
    MPI_Type_commit(&subgrid);

    MPI_Type_vector(local_H+2, 1, local_W+2, MPI_REAL_T, &column);
    MPI_Type_vector(1, local_W+2, local_W+2, MPI_REAL_T, &row);

    MPI_Type_commit(&column);
    MPI_Type_commit(&row);

    MPI_Type_create_hvector(6, 1, (local_W+2)*(local_H+2)*sizeof(real_t), column, &columns);
    MPI_Type_create_hvector(6, 1, (local_W+2)*(local_H+2)*sizeof(real_t), row, &rows);

    MPI_Type_commit(&columns);
    MPI_Type_commit(&rows);
}

void init_domain(void)
{
    float radius = H / 20.0;
    int center[2] = { H/2, W/4 };

    int local_offset[2] = { cart_pos[0] * local_H, cart_pos[1] * local_W };

    for (int i = 0; i < local_H+2; i++) {
        for (int j = 0; j <= local_W+2; j++) {
            bool in_circle = radius >
                sqrt((i-1+local_offset[0]-center[0]) * (i-1+local_offset[0]-center[0]) +
                     (j-1+local_offset[1]-center[1]) * (j-1+local_offset[1]-center[1]));
            if (in_circle) {
                LATTICE(i,j) = SOLID;
            } else {
                LATTICE(i,j) = FLUID;
            }
        }
    }

    // Fill top wall
    if (cart_pos[0] == 0) {
        for (int j = 0 ; j < local_W+2; j++) {
            LATTICE(1, j) = WALL;
        }
    }

    // Fill bottom wall
    if (cart_pos[0] == dims[0] - 1) {
        for (int j = 0 ; j < local_W+2; j++) {
            LATTICE(local_H, j) = WALL;
        }
    }

    for (int i = 1; i <= local_H; i++) {
        for (int j = 1; j <= local_W; j++) {
            if (LATTICE(i,j) == SOLID) {
                for (int d = 0; d < DIRECTIONS-1; d++) {
                    int ni = (i + OFFSETS[i%2][d][0]+H)%H;
                    int nj = (j + OFFSETS[i%2][d][1]+W)%W;

                    /*if (ni < 0 || ni >= local_H+2 || nj < 0 || nj >= local_W+2)*/
                    /*    continue;*/

                    if (LATTICE(ni, nj) == FLUID)
                        LATTICE(i,j) = WALL;
                }
            }
        }
    }

}


void border_exchange(void) {
    // Send north
    MPI_Sendrecv(&D_nxt(1, 0, 0), 1, rows, cart_nbo[NORTH], 0,
                 &D_nxt(local_H+1, 0, 0), 1, rows, cart_nbo[SOUTH], 0,
                 comm_cart, MPI_STATUS_IGNORE);

    // Send south
    MPI_Sendrecv(&D_nxt(local_H, 0, 0), 1, rows, cart_nbo[SOUTH], 1,
                 &D_nxt(0, 0, 0), 1, rows, cart_nbo[NORTH], 1,
                 comm_cart, MPI_STATUS_IGNORE);

    // Send west
    MPI_Sendrecv(&D_nxt(0, 1, 0), 1, columns, cart_nbo[WEST], 2,
                 &D_nxt(0, local_W+1, 0), 1, columns, cart_nbo[EAST], 2,
                 comm_cart, MPI_STATUS_IGNORE);

    // Send east
    MPI_Sendrecv(&D_nxt(0, local_W, 0), 1, columns, cart_nbo[EAST], 3,
            &D_nxt(0, 0, 0), 1, columns, cart_nbo[WEST], 3,
            comm_cart, MPI_STATUS_IGNORE);
}

void collide(void)
{
    #pragma omp parallel for
    for (int i = 1; i <= local_H; i++) {
        for (int j = 1; j <= local_W; j++) {
            real_t rho      = 0.0;  // Density
            real_t ev       = 0.0;  // Dot product of e and v;
            real_t N_eq     = 0.0;  // Equilibrium at i
            real_t delta_N  = 0.0;  // Change

            // Ignore solid sites
            if (LATTICE(i,j) == SOLID) {
                continue;
            }
            rho = 0.0;
            V_x(i,j) = V_y(i,j) = 0.0;
            if (LATTICE(i,j) == FLUID) {
                for (int d = 0; d < DIRECTIONS; d++) {
                    rho += D_now(i,j,d);
                    V_y(i,j) += e[d][0] * D_now(i,j,d);
                    V_x(i,j) += e[d][1] * D_now(i,j,d);
                }
                V_y(i,j) /= rho;
                V_x(i,j) /= rho;
            }

            for (int d = 0; d < DIRECTIONS; d++) {
                // Boundary condition: Reflect of walls
                if (LATTICE(i,j) == WALL) {
                    if (d != 6) {
                        D_nxt(i,j,(d+3)%6) = D_now(i,j,d);
                    }
                    continue;
                }

                ev = e[d][1] * V_x(i,j) + e[d][0] * V_y(i,j);
                if (d == 6) {
                    // Rest particle
                    N_eq = ALPHA*rho - rho*(V_x(i,j)*V_x(i,j) + V_y(i,j)*V_y(i,j));
                } else {
                    // Outgoing vectors
                    N_eq =
                        // F_eq_i
                        rho*(1.0-ALPHA)/6.0
                        + rho/3.0*ev
                        + (2.0*rho/3.0)*ev*ev
                        - rho/6.0*(V_x(i,j)*V_x(i,j)+V_y(i,j)*V_y(i,j));
                }

                delta_N = -(D_now(i,j,d)-N_eq)/TAU;

                if (cart_pos[1] * local_W + j == 1)
                    delta_N += (1.0/3.0) * force[1] * e[d][1];

                D_nxt(i,j,d) = D_now(i,j,d) + delta_N;
            }

        }
    }
}

void stream(void)
{
    #pragma omp parallel for
    for (int i = 0; i < local_H+2; i++) {
        for (int j = 0; j < local_W+2; j++) {
            for (int d = 0; d < DIRECTIONS; d++) {
                int ni = (i + OFFSETS[i%2][d][0]);
                int nj = (j + OFFSETS[i%2][d][1]);

                if (ni < 0 || ni >= local_H+2 || nj < 0 || nj >= local_W+2)
                    continue;

                D_now(ni,nj,d) = D_nxt(i,j,d);
            }
        }
    }
}

void save(int iteration)
{
    // Caculate absolute velocity (without halo)
    for (int i = 1; i <= local_H; i++) {
        for (int j = 1; j <= local_W; j++) {
            OUTBUF((i-1),(j-1)) = (float) sqrt(V_y(i,j)*V_y(i,j) + V_x(i,j)*V_x(i,j));
        }
    }

    char filename[256];
    memset(filename, 0, 256);
    sprintf(filename, "data/%05d.dat", iteration);

    MPI_File output = NULL;
    MPI_File_open(comm_cart, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &output);

    if (!output) {
        fprintf(stderr, "WARNING: Unable to open file '%s'. Did not save iteration %d\n", filename, iteration);
        exit(EXIT_FAILURE);
    }

    MPI_File_set_view(output, 0, MPI_FLOAT, subgrid, "native", MPI_INFO_NULL);
    MPI_File_write_all(output, outbuf, local_H*local_W, MPI_FLOAT, MPI_STATUS_IGNORE);

    MPI_File_close(&output);
}

void options(int argc, char **argv)
{
    timesteps = 100;
    store_freq = 1;
    H = 1600;
    W = 2400;

    int c;
    while ((c = getopt(argc, argv, "i:s:h")) != -1 ) {
        switch (c) {
            case 'i':
                timesteps = strtol(optarg, NULL, 10);
                break;
            case 's':
                store_freq = strtol(optarg, NULL, 10);
                printf("store_freq=%d\n", store_freq);
                break;
            case 'h':
                printf("Usage: d2q7 [-i iter]\n");
                printf("  options\n");
                printf("    -i iter  number of iterations (default 40000)\n");
                printf("    -s freq  store frequency      (default 100)\n");
                printf("    -h       display this message\n");
                exit(EXIT_SUCCESS);
                break;
            default:
                opterr = 0;
                fprintf(stderr, "ERROR: Illegal argument %c\n", c);
                exit(EXIT_FAILURE);
                break;
        }
    }
}
