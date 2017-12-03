#include <x86intrin.h>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <mpi.h>

/// store number of grid points in one dimension
std::size_t grid_points_1d = 0;
/// Number of processes per dimension
std::size_t procs_1d = 0;

/// Number of processes in MPI_COMM_WORLD
int world_size;

// Cartesian world communicator
MPI_Comm cartcomm;

// Number of dimensions
int n_dims = 2;

// Cartesian world rank
int cart_rank;

// Vector datatype for sending data to other processes
MPI_Datatype vector_datatype;

// Cartesian coordinates
int coords[2] = {0, 0};

// boundary points for Dirichlet conditions
int _istart = 0, _jstart = 0, _iend = GRID_SIZE, _jend = GRID_SIZE;

/// store begin timestep
struct timeval begin;
/// store end timestep
struct timeval end;

/**
 * initialize and start timer
 */
void timer_start()
{
	gettimeofday(&begin,(struct timezone *)0);
}

/**
 * stop timer and return measured time
 *
 * @return measured time
 */
double timer_stop()
{
	gettimeofday(&end,(struct timezone *)0);
	double seconds, useconds;
	double ret, tmp;

	if (end.tv_usec >= begin.tv_usec)
	{
		seconds = (double)end.tv_sec - (double)begin.tv_sec;
		useconds = (double)end.tv_usec - (double)begin.tv_usec;
	}
	else
	{
		seconds = (double)end.tv_sec - (double)begin.tv_sec;
		seconds -= 1;					// Correction
		useconds = (double)end.tv_usec - (double)begin.tv_usec;
		useconds += 1000000;			// Correction
	}

	// get time in seconds
	tmp = (double)useconds;
	ret = (double)seconds;
	tmp /= 1000000;
	ret += tmp;

	return ret;
}

/**
 * stores a given grid into a file
 * 
 * @param grid the grid that should be stored
 * @param filename the filename
 */
void store_grid(double* grid, std::string filename)
{
	std::fstream filestr;
	filestr.open (filename.c_str(), std::fstream::out);
	
	// calculate mesh width 
	double mesh_width = 1.0/((double)(grid_points_1d-1));

	// store grid incl. boundary points
	for (int i = 0; i < grid_points_1d; i++)
	{
		for (int j = 0; j < grid_points_1d; j++)
		{
			filestr << mesh_width*i << " " << mesh_width*j << " " << grid[(i*grid_points_1d)+j] << std::endl;
		}
		
		filestr << std::endl;
	}

	filestr.close();
}

void fill_grid(double* o_grid, double* grid, int* c) {
	int offset = (c[0] * GRID_SIZE * grid_points_1d) + (c[1] * GRID_SIZE);
	for (int i=0; i<GRID_SIZE; i++) {
		for (int j=0; j<GRID_SIZE; j++) {
			o_grid[offset + ((i*grid_points_1d) + j)] = grid[(i*GRID_SIZE) + j];
		}
	}
}

void print_grid(double* o_grid, std::size_t size) {
	printf("printing grid\n");
	for (int i=0; i<size; i++) {
		for (int j=0; j<size; j++) {
			printf("%2d=%0.2f  ", (i*size) + j, o_grid[(i*size) + j]);
		}
		printf("\n");
	}
	printf("end printing grid\n");
}

double* collect_grid(double* grid) {
	if (cart_rank == 0) {
		double* o_grid = (double*)_mm_malloc(grid_points_1d*grid_points_1d*sizeof(double), 64);
		double* in_grid = (double*)_mm_malloc(GRID_SIZE*GRID_SIZE*sizeof(double), 64);
		fill_grid(o_grid, grid, coords);
		for (int i=1; i<world_size; i++) {
			MPI_Recv(in_grid, GRID_SIZE*GRID_SIZE, 
				MPI_DOUBLE, i, 0,
	             cartcomm, MPI_STATUS_IGNORE);
			int c[2];
			MPI_Cart_coords(cartcomm, i, n_dims, c);
			fill_grid(o_grid, in_grid, c);
		}
		return o_grid;
	} else {
		MPI_Send(grid, GRID_SIZE*GRID_SIZE, MPI_DOUBLE, 0, 0,
             cartcomm);
		return NULL;
	}
}


void init_indexes() {
	if (coords[0] == 0) {
		_istart = 1;
	}
	if (coords[0] == procs_1d-1) {
		_iend = GRID_SIZE-1;
	}
	if (coords[1] == 0) {
		_jstart = 1;
	}
	if (coords[1] == procs_1d-1) {
		_jend = GRID_SIZE-1;
	}
}

/**
 * calculate the grid's initial values for given grid points
 *
 * @param x the x-coordinate of a given grid point
 * @param y the y-coordinate of a given grid point
 *
 * @return the initial value at position (x,y)
 */
double eval_init_func(double x, double y)
{
	return (x*x)*(y*y);
}

/**
 * initializes a given grid: inner points are set to zero
 * boundary points are initialized by calling eval_init_func
 *
 * @param grid the grid to be initialized
 */
void init_grid(double* grid)
{
	// set all points to zero
	// If it is an extreme row or column
	// Stored in row major order
	
	for (int i = 0; i < GRID_SIZE*GRID_SIZE; i++)
	{
		grid[i] = 0.0;
	}
	
	if (coords[0] == 0 || coords[0] == procs_1d-1 ||
		coords[1] == 0 || coords[1] == procs_1d-1) 
	{
		double mesh_width = 1.0/((double)(grid_points_1d-1));

		for (int i = 0; i < GRID_SIZE; i++) 
		{
			double m = (coords[0] * GRID_SIZE) + i;
			double n = (coords[1] * GRID_SIZE) + i;
			// x-boundaries only set for extreme cols
			// For first column
			if (coords[1] == 0) {
				grid[i*GRID_SIZE] = eval_init_func(0.0, m*mesh_width);
			}
			// For last column
			if (coords[1] == procs_1d-1) {
				grid[(i*GRID_SIZE) + GRID_SIZE-1] = eval_init_func(1.0, m*mesh_width);
			}
			
			// y-boundaries only set for extreme rows
			// For first row
			if (coords[0] == 0) {
				grid[i] = eval_init_func(n*mesh_width, 0.0);
			}
			// For last row
			if (coords[0] == procs_1d-1) {
				grid[((GRID_SIZE-1)*GRID_SIZE) + i] = eval_init_func(n*mesh_width, 1.0);
			}
		}
	}
}

/**
 * initializes the righ * solve the Laplace equation instead of Poisson (-> b=0)
t hand side, we want to keep it simple and
 *
 * @param b the right hand side
 */
void init_b(double* b)
{
	// set all points to zero
	for (int i = 0; i < GRID_SIZE*GRID_SIZE; i++)
	{
		b[i] = 0.0;
	}
}

/**
 * copies data from one grid to another
 *
 * @param dest destination grid
 * @param src source grid
 */
void g_copy(double* dest, double* src)
{
	for (int i = 0; i < GRID_SIZE*GRID_SIZE; i++)
	{
		dest[i] = src[i];
	}
}

/**
 * calculates the dot product of the two grids (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 *
 * @param grid1 first grid
 * @param grid2 second grid
 */
double g_dot_product(double* grid1, double* grid2)
{
	double tmp = 0.0;

	for (int i = _istart; i < _iend; i++)
	{
		for (int j = _jstart; j < _jend; j++)
		{
			tmp += (grid1[(i*GRID_SIZE)+j] * grid2[(i*GRID_SIZE)+j]);
		}
	}
	
	return tmp;
}

/**
 * scales a grid by a given scalar (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 *
 * @param grid grid to be scaled
 * @param scalar scalar which is used to scale to grid
 */
void g_scale(double* grid, double scalar)
{
	for (int i = _istart; i < _iend; i++)
	{
		for (int j = _jstart; j < _jend; j++)
		{
			grid[(i*GRID_SIZE)+j] *= scalar;
		}
	}
}

/**
 * implements BLAS's Xaxpy operation for grids (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 *
 * @param dest destination grid
 * @param src source grid
 * @param scalar scalar to scale to source grid
 */
void g_scale_add(double* dest, double* src, double scalar)
{
	for (int i = _istart; i < _iend; i++)
	{
		for (int j = _jstart; j < _jend; j++)
		{
			dest[(i*GRID_SIZE)+j] += (scalar*src[(i*GRID_SIZE)+j]);
		}
	}
}

/**
 * implements the the 5-point finite differences stencil (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 * 
 * @param grid grid for which the stencil should be evaluated
 * @param result grid where the stencil's evaluation should be stored
 * @param n_data data from the neighbour for applying 5 point on boundaries
 *			size is 4 * GRID_SIZE
 * 			indexes :0(left) 1(right) 2(top) 3(bottom)
 *			This data is filled up by this function. 
 * 			Only the memory is allocated in the parent.
 */
void g_product_operator(double* grid, 
						double* result,
						double* n_data)
{
	double mesh_width = 1.0/((double)(grid_points_1d-1));

	// Source rank is the same as cart_rank
	int src_rank, l_rank, r_rank, t_rank, b_rank;
	l_rank = r_rank = t_rank = b_rank = -1;
	MPI_Request request[4][2];
	bool request_valid[4] = {false, false, false, false};
	
	// If I don't have a neighbour I neither send to him nor recieve from him.
	if (_jstart != 1) {
		MPI_Cart_shift(cartcomm, 1, -1, &src_rank, &l_rank);
		// Left guy exists send to him
		// Columns are stored non-contiguously. Send first column
		MPI_Isend(grid,
					1, vector_datatype, l_rank, 0, cartcomm, &(request[0][0]));
	  	MPI_Irecv(n_data, GRID_SIZE, MPI_DOUBLE, l_rank, 0, cartcomm, &(request[0][1]));
	  	request_valid[0] = true;
	}
	if (_jend != GRID_SIZE-1) {
		MPI_Cart_shift(cartcomm, 1,  1, &src_rank, &r_rank);
		// Right guy exists send to him
		// Columns are stored non-contiguously. Send last column
		MPI_Isend(grid + GRID_SIZE-1,
					1, vector_datatype, r_rank, 0, cartcomm, &(request[1][0]));
	  	MPI_Irecv(n_data + GRID_SIZE,
	  				GRID_SIZE, MPI_DOUBLE, r_rank, 0, cartcomm, &(request[1][1]));
	  	request_valid[1] = true;
	}
	if (_istart != 1) {
		MPI_Cart_shift(cartcomm, 0, -1, &src_rank, &t_rank);
		// Rows are stored contiguously. Send first row
		MPI_Isend(grid,
					GRID_SIZE, MPI_DOUBLE, t_rank, 0, cartcomm, &(request[2][0]));
	  	MPI_Irecv(n_data + (2 * GRID_SIZE), 
	  				GRID_SIZE, MPI_DOUBLE, t_rank, 0, cartcomm, &(request[2][1]));
	  	request_valid[2] = true;
	}
	if (_iend != GRID_SIZE-1) {
		MPI_Cart_shift(cartcomm, 0,  1, &src_rank, &b_rank);
		// Rows are stored contiguously. Send last row
		MPI_Isend(grid + (GRID_SIZE * (GRID_SIZE-1)),
					GRID_SIZE, MPI_DOUBLE, b_rank, 0, cartcomm, &(request[3][0]));
	  	MPI_Irecv(n_data + (3 * GRID_SIZE), 
	  				GRID_SIZE, MPI_DOUBLE, b_rank, 0, cartcomm, &(request[3][1]));
	  	request_valid[3] = true;
	}

	// Wait for all requests to complete
	for (int i=0; i<4; i++) {
		int target_pid = -1;
		char *pos;
		char *left = "left";
		char *right = "right";
		char *top = "top";
		char *bottom = "bottom";

		if (i==0) {
			pos = left;
			target_pid = l_rank;
		}
		if (i==1) {
			pos = right;
			target_pid = r_rank;
		}
		if (i==2) {
			pos = top;
			target_pid = t_rank;
		}
		if (i==3) {
			pos = bottom;
			target_pid = b_rank;
		}
		if (request_valid[i]) {
			MPI_Wait(&(request[i][0]), MPI_STATUS_IGNORE);
			MPI_Wait(&(request[i][1]), MPI_STATUS_IGNORE);
		}
	}

	for (int i = _istart; i < _iend; i++)
	{
		for (int j = _jstart; j < _jend; j++)
		{
			double left, right, top, bottom;

			if ((i+1) >= GRID_SIZE) {
				bottom = n_data[(3*GRID_SIZE) + j];
			} else {
				bottom = grid[((i+1)*GRID_SIZE)+j];
			}
			if ((i-1) < 0) {
				top = n_data[(2*GRID_SIZE) + j];
			} else {
				top = grid[((i-1)*GRID_SIZE)+j];
			}
			if ((j+1) >= GRID_SIZE) {
				right = n_data[(1*GRID_SIZE)+i];
			} else {
				right = grid[(i*GRID_SIZE)+j+1];
			}
			if ((j-1) < 0) {
				left = n_data[i];
			} else {
				left = grid[(i*GRID_SIZE)+j-1];
			}

			result[(i*GRID_SIZE)+j] =  (
							(4.0*grid[(i*GRID_SIZE)+j]) 
							- bottom
							- top
							- right
							- left
							) * (mesh_width*mesh_width);
		}
	}
}

/**
 * The CG Solver (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 *
 * For details please see :
 * http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
 *
 * @param grid the grid containing the initial condition
 * @param b the right hand side
 * @param cg_max_iterations max. number of CG iterations 
 * @param cg_eps the CG's epsilon
 */
std::size_t solve(double* grid, double* b, std::size_t cg_max_iterations, double cg_eps)
{

	// std::cout << "Starting Conjugated Gradients for process x:" << cart_rank<<std::endl;

	double eps_squared = cg_eps*cg_eps;
	std::size_t needed_iters = 0;

	// define temporal vectors
	std::size_t size = GRID_SIZE*GRID_SIZE;
	double* q = (double*)_mm_malloc(size*sizeof(double), 64);
	double* r = (double*)_mm_malloc(size*sizeof(double), 64);
	double* d = (double*)_mm_malloc(size*sizeof(double), 64);
	double* b_save = (double*)_mm_malloc(size*sizeof(double), 64);

	g_copy(q, grid);
	g_copy(r, grid);
	g_copy(d, grid);
	g_copy(b_save, b);
	
	double delta_0 = 0.0;
	double delta_old = 0.0;
	double delta_new = 0.0;
	double beta = 0.0;
	double a = 0.0;
	double residuum = 0.0;

	// Setup neighbours and their data
	// Do this once here so that g_product operator doesn't need to care about it.
	double* n_grid_data = (double*)_mm_malloc(GRID_SIZE*4*sizeof(double), 64);
	double* n_d_data = (double*)_mm_malloc(GRID_SIZE*4*sizeof(double), 64);

	g_product_operator(grid, d, n_grid_data);
	g_scale_add(b, d, -1.0);
	g_copy(r, b);
	g_copy(d, r);

	// calculate starting norm
	double r_reduce, r_dot = g_dot_product(r, r);
	MPI_Allreduce(&r_dot, &r_reduce, 1,
                  MPI_DOUBLE, MPI_SUM, cartcomm);
	delta_new = r_reduce;
	delta_0 = delta_new*eps_squared;
	residuum = (delta_0/eps_squared);
	if (cart_rank == 0) {
		std::cout << "Starting norm of residuum: " << (delta_0/eps_squared) << std::endl;
		std::cout << "Target norm:               " << (delta_0) << std::endl;	
	}

	while ((needed_iters < cg_max_iterations) && (delta_new > delta_0))
	{
		// q = A*d
		g_product_operator(d, q, n_d_data);

		// a = d_new / d.q
		// Reduce results of dot product from all processes
		r_dot = g_dot_product(d, q);
		MPI_Allreduce(&r_dot, &r_reduce, 1,
					MPI_DOUBLE, MPI_SUM, cartcomm);
		a = delta_new/r_reduce;
		
		// x = x + a*d
		g_scale_add(grid, d, a);
		
		if ((needed_iters % 50) == 0)
		{
			g_copy(b, b_save);
			g_product_operator(grid, q, n_grid_data);
			g_scale_add(b, q, -1.0);
			g_copy(r, b);
		}
		else
		{
			// r = r - a*q
			g_scale_add(r, q, -a);
		}

		// calculate new deltas and determine beta
		r_dot = g_dot_product(r, r);
		MPI_Allreduce(&r_dot, &r_reduce, 1,
					MPI_DOUBLE, MPI_SUM, cartcomm);
		delta_old = delta_new;
		delta_new = r_reduce;
		beta = delta_new/delta_old;

		// adjust d
		g_scale(d, beta);
		g_scale_add(d, r, 1.0);
		
		residuum = delta_new;
		needed_iters++;
		// if (cart_rank == 0) {
		// 	std::cout << "(iter: " << needed_iters << ")delta: " << delta_new << std::endl;
		// }
	}
	if (cart_rank == 0) {
		std::cout << "Number of iterations: " << needed_iters << " (max. " << cg_max_iterations << ")" << std::endl;
		std::cout << "Final norm of residuum: " << delta_new << std::endl;	
	}	
	
	_mm_free(n_grid_data);
	_mm_free(n_d_data);
	_mm_free(d);
	_mm_free(q);
	_mm_free(r);
	_mm_free(b_save);
}

/**
 * main application
 *
 * @param argc number of cli arguments
 * @param argv values of cli arguments
 */
int main(int argc, char* argv[])
{
	// check if all parameters are specified
	if (argc != 4)
	{
		std::cout << std::endl;
		std::cout << "meshwidth" << std::endl;
		std::cout << "cg_max_iterations" << std::endl;
		std::cout << "cg_eps" << std::endl;
		std::cout << std::endl;
		std::cout << "example:" << std::endl;
		std::cout << "./app 0.125 100 0.0001" << std::endl;
		std::cout << std::endl;
		
		return -1;
	}
	
	// read cli arguments
	double mesh_width = atof(argv[1]);
	size_t cg_max_iterations = atoi(argv[2]);
	double cg_eps = atof(argv[3]);

	// calculate grid points per dimension
	// grid_points_1d = (std::size_t)(1.0/mesh_width)+1;
	grid_points_1d = (std::size_t)(1.0/mesh_width);

	/**
	 * MPI intiliaze stuff
	**/
	MPI_Init( NULL, NULL );
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	if (grid_points_1d % GRID_SIZE != 0 ) {
		fprintf( stderr, 
			"grid_size (%d) must be multiple of process grid size (%d).\n",
			grid_points_1d, GRID_SIZE);
		exit(1);
	}
	procs_1d = grid_points_1d/GRID_SIZE;
	
	if (world_size != procs_1d*procs_1d) {
		fprintf( stderr, 
			"world size must be a square of %s i.e. %d as per current conf\n",
			"grid_size/grid_size_per_process", procs_1d * procs_1d);
		exit(1);
	}

	int dims[2] = {procs_1d , procs_1d};
	int periods[2] = {1, 1}; // Periodic
	MPI_Cart_create(MPI_COMM_WORLD, n_dims, dims, periods, 1, &cartcomm);
	MPI_Comm_rank(cartcomm, &cart_rank);
	MPI_Cart_coords(cartcomm, cart_rank, n_dims, coords);
	
	// All processes have the same initial state
	// initialize the grid and rights hand side
	double* grid = (double*)_mm_malloc(
		GRID_SIZE*GRID_SIZE*sizeof(double), 
		64);
	double* b = (double*)_mm_malloc(
		GRID_SIZE*GRID_SIZE*sizeof(double), 64);
	init_grid(grid);
	// TODO: Need to collect grid before it can be stored. 
	// store_grid(grid, "initial_condition.gnuplot");
	init_b(b);
	// store_grid(b, "b.gnuplot");
	init_indexes(); // so as to not modify boundary points acc to Dirichlet conditions
	
	// Initialize vector data type to send non contiguous data
	MPI_Type_vector(GRID_SIZE, 1, GRID_SIZE, MPI_DOUBLE, &vector_datatype);
	MPI_Type_commit(&vector_datatype);

	/// TEST BLOCK
	/// TEST BLOCK

	// solve Poisson equation using CG method
	timer_start();
	solve(grid, b, cg_max_iterations, cg_eps);
	double time = timer_stop();
	// Now print it
	double* o_grid = NULL;
	// o_grid = collect_grid(grid);
	if (cart_rank == 0) {
		std::cout << std::endl << "Needed time: " << time << " s" << std::endl << std::endl;
		// print_grid(o_grid, grid_points_1d);
		// printf("Starting storage\n");
		// store_grid(o_grid, "solution.gnuplot");
		printf("Starting cleanup\n");
	}
	if (o_grid != NULL) _mm_free(o_grid);
	MPI_Type_free(&vector_datatype);
	MPI_Comm_free(&cartcomm);
	_mm_free(grid);
	_mm_free(b);
	if (cart_rank == 0)
		printf("Finished cleanup. Finalizing\n");
	MPI_Finalize();

	return 0;
}