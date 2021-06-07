#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define Told(i, j) Told[(N*(i) + (j))] //row i, col j
#define Tnew(i, j) Tnew[(N*(i) + (j))] //row i, col j


void dump(double* data, int n, char* fname){
	FILE* fp = fopen(fname,"wb");
    fwrite(data, sizeof(double), n, fp);
    fclose(fp);

}
    

void printMatrix(double* data, int row, int col){
    for(int i=0; i<row; i++){
        if (i==0){
        	printf("[[");
        }
        else{
            printf(" [");
        }
        for(int j=0; j<col; j++){
            printf("%9.4g,", data[col*i + j]);
        }
        if (i!=row-1){
            printf("]\n");
        }
        else{
            printf("]]\n");
        }
    }
}

void dump_to_file(double* restrict Told, int N, int M, int nx, int ny, int niter, double L, double H, double dt, int rank, int poolsize, int nrow, int ncol, MPI_Comm ComArray, char* fname){
	if (rank != 0){
		MPI_Send(Told, M*N, MPI_DOUBLE, 0, rank, ComArray);
		return;
	}

	double* T = malloc((nx+1)*(ny+1)*sizeof(double));
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			T[i*(nx+1) + j] = Told(i,j);
		}
	}

	for(int r=1; r < poolsize; r++){
		int coords[2];
		MPI_Cart_coords(ComArray, r, 2, coords);
		int row = coords[0];
		int col = coords[1];

		int jmin = (nx*col)/ncol;
		int jmax = (nx*(col+1))/ncol;
		int N = jmax-jmin+1; //local 

		int imin = (ny*row)/nrow;
		int imax = (ny*(row+1))/nrow;
		int M = imax-imin+1;
		double* Tbuff = malloc(M*N*sizeof(double));

		MPI_Recv(Tbuff, M*N, MPI_DOUBLE, r, r, ComArray, MPI_STATUS_IGNORE);
		
		for(int i = 0; i < M; i++){
			for(int j = 0; j < N; j++){
				T[(i + imin)*(nx+1) + j + jmin] = Tbuff[i*N+j];
			}
		}
		free(Tbuff);
	}

	//printMatrix(T, ny+1, nx+1);

    FILE* fp = fopen(fname,"wb");

    fwrite(&nx, sizeof(int), 1, fp);
    fwrite(&ny, sizeof(int), 1, fp);
	fwrite(&L, sizeof(double), 1, fp);
	fwrite(&H, sizeof(double), 1, fp);
	fwrite(&dt, sizeof(double), 1, fp);
    fwrite(&niter, sizeof(int), 1, fp);
    fwrite(T, sizeof(double), (nx+1)*(ny+1), fp);
    fclose(fp);

	free(T);
}

//makes sure that the shape of the domain is aligned with the shape of the process array
void reorder_dims(int dims[], int nx, int ny){
	int dmin = dims[0] < dims[1] ? dims[0] : dims[1];
	int dmax = dims[0] > dims[1] ? dims[0] : dims[1];
	if (nx > ny){
		dims[0] = dmin;
		dims[1] = dmax;
	}
	else{
		dims[0] = dmax;
		dims[1] = dmin;
	}
}

void i_recieve(double* boundary_data[], int neighbours[], int M, int N, MPI_Request recv_requests[], MPI_Comm ComArray){
	int up = neighbours[0];
	int down = neighbours[1];
	int left = neighbours[2];
	int right = neighbours[3];

	int tag = 0;
	if(up >= 0){
		MPI_Irecv(boundary_data[0], N, MPI_DOUBLE, up, tag, ComArray, &recv_requests[0]);
	}
	if(down >= 0){
		MPI_Irecv(boundary_data[1], N, MPI_DOUBLE, down, tag, ComArray, &recv_requests[1]);
	}
	if(left >= 0){
		MPI_Irecv(boundary_data[2], M, MPI_DOUBLE, left, tag, ComArray, &recv_requests[2]);
	}
	if(right >= 0){
		MPI_Irecv(boundary_data[3], M, MPI_DOUBLE, right, tag, ComArray, &recv_requests[3]);
	}
}

void wait_recieve(int neighbours[], MPI_Request recv_requests[]){
	int up = neighbours[0];
	int down = neighbours[1];
	int left = neighbours[2];
	int right = neighbours[3];

	int tag = 0;
	if(up >= 0){
		MPI_Wait(&recv_requests[0], MPI_STATUS_IGNORE);
	}
	if(down >= 0){
		MPI_Wait(&recv_requests[1], MPI_STATUS_IGNORE);
	}
	if(left >= 0){
		MPI_Wait(&recv_requests[2], MPI_STATUS_IGNORE);
	}
	if(right >= 0){
		MPI_Wait(&recv_requests[3], MPI_STATUS_IGNORE);
	}
}


void i_send(double* Told, int neighbours[], int M, int N, MPI_Request send_requests[], MPI_Datatype ColDtype, MPI_Comm ComArray){
	int up = neighbours[0];
	int down = neighbours[1];
	int left = neighbours[2];
	int right = neighbours[3];

	int tag = 0;
	if(up >= 0){
		MPI_Isend(&Told(1,0), N, MPI_DOUBLE, up, tag, ComArray, &send_requests[0]);
	}
	if(down >= 0){
		MPI_Isend(&Told(M-2,0), N, MPI_DOUBLE, down, tag, ComArray, &send_requests[1]);
	}
	if(left >= 0){
		MPI_Isend(&Told(0,1), 1, ColDtype, left, tag, ComArray, &send_requests[2]);
	}
	if(right >= 0){
		MPI_Isend(&Told(0,N-2), 1, ColDtype, right, tag, ComArray, &send_requests[3]);
	}
}

void wait_send(int neighbours[], MPI_Request send_requests[]){
	int up = neighbours[0];
	int down = neighbours[1];
	int left = neighbours[2];
	int right = neighbours[3];

	int tag = 0;
	if(up >= 0){
		MPI_Wait(&send_requests[0], MPI_STATUS_IGNORE);
	}
	if(down >= 0){
		MPI_Wait(&send_requests[1], MPI_STATUS_IGNORE);
	}
	if(left >= 0){
		MPI_Wait(&send_requests[2], MPI_STATUS_IGNORE);
	}
	if(right >= 0){
		MPI_Wait(&send_requests[3], MPI_STATUS_IGNORE);
	}
}


void compute_inside(double* restrict Told, double* restrict Tnew, double dx, double dy, double kdt, int M, int N){
	double dx2 = dx*dx;
	double dy2 = dy*dy;
	
	for(int i = 2; i < M-2; i++){
		for(int j = 2; j < N-2; j++){
			Tnew(i,j) = Told(i,j) + kdt*(  (Told(i,j+1) - 2*Told(i,j) + Told(i,j-1))/dx2 + (Told(i+1,j) - 2*Told(i,j) + Told(i-1,j))/dy2);
		}
	}
}


void internal_bounds(double* restrict Told, double* restrict Tnew, double dx, double dy, double kdt, int M, int N){
	double dx2 = dx*dx;
	double dy2 = dy*dy;
	
	for(int j = 1; j < N-1; j++){
		Tnew(1,j) = Told(1,j) + kdt*(  (Told(1,j+1) - 2*Told(1,j) + Told(1,j-1))/dx2 + (Told(1+1,j) - 2*Told(1,j) + Told(1-1,j))/dy2);
	}
	for(int j = 1; j < N-1; j++){
		Tnew(M-2,j) = Told(M-2,j) + kdt*(  (Told(M-2,j+1) - 2*Told(M-2,j) + Told(M-2,j-1))/dx2 + (Told(M-2+1,j) - 2*Told(M-2,j) + Told(M-2-1,j))/dy2);
	}
	for(int i = 2; i < M-2; i++){
		Tnew(i,1) = Told(i,1) + kdt*(  (Told(i,1+1) - 2*Told(i,1) + Told(i,1-1))/dx2 + (Told(i+1,1) - 2*Told(i,1) + Told(i-1,1))/dy2);
	}
	for(int i = 2; i < M-2; i++){
		Tnew(i,N-2) = Told(i,N-2) + kdt*(  (Told(i,N-2+1) - 2*Told(i,N-2) + Told(i,N-2-1))/dx2 + (Told(i+1,N-2) - 2*Told(i,N-2) + Told(i-1,N-2))/dy2);
	}
}

void compute_boundary(double* restrict Told, double* restrict Tnew, double* boundary_data[], int M, int N, double (*f)(double, double), double dx, double dy, int jmin, int imin, double kdt){
	Tnew(0,0) = Told(0,0);
	Tnew(M-1,0) = Told(M-1,0);
	Tnew(0,N-1) = Told(0,N-1);
	Tnew(M-1,N-1) = Told(M-1,N-1);

	double dx2 = dx*dx;
	double dy2 = dy*dy;
	
	if (boundary_data[0] != NULL){
		double* ext = boundary_data[0];
		for (int j=1; j < N-1; j++){
			Tnew(0, j) = Told(0, j) +  kdt*( (Told(0,j+1) - 2*Told(0,j) + Told(0,j-1))/dx2 + (Told(1,j) - 2*Told(0,j) + ext[j])/dy2);
		}
		Tnew(0,0) += kdt*(Told(1,0) - 2*Told(0,0) + ext[0])/dy2;
		Tnew(0,N-1) += kdt*(Told(1,N-1) - 2*Told(0,N-1) + ext[N-1])/dy2;
	}

	if (boundary_data[1] != NULL){
		double* ext = boundary_data[1];
		for (int j=1; j < N-1; j++){
			Tnew(M-1, j) = Told(M-1, j) +  kdt*( (Told(M-1,j+1) - 2*Told(M-1,j) + Told(M-1,j-1))/dx2 + (ext[j] - 2*Told(M-1,j) + Told(M-2,j))/dy2);
		}
		Tnew(M-1,0) += kdt*(ext[0] - 2*Told(M-1,0) + Told(M-2,0))/dy2;
		Tnew(M-1,N-1) += kdt*(ext[N-1] - 2*Told(M-1,N-1) + Told(M-2,N-1))/dy2;
	}
	
	if (boundary_data[2] != NULL){
		double* ext = boundary_data[2];
		for (int i=1; i < M-1; i++){
			Tnew(i,0) = Told(i,0) + kdt*(  (Told(i,0+1) - 2*Told(i,0) + ext[i])/dx2 + (Told(i+1,0) - 2*Told(i,0) + Told(i-1,0))/dy2);
		}
		Tnew(0,0) += kdt*(Told(0,1) - 2*Told(0,0) + ext[0])/dx2;
		Tnew(M-1,0) += kdt*(Told(M-1,1) - 2*Told(M-1,0) + ext[M-1])/dx2;
	}

	if (boundary_data[3] != NULL){
		double* ext = boundary_data[3];
		for (int i=1; i < M-1; i++){
			Tnew(i,N-1) = Told(i,N-1) + kdt*(  (ext[i] - 2*Told(i,N-1) + Told(i,N-2))/dx2 + (Told(i+1,N-1) - 2*Told(i,N-1) + Told(i-1,N-1))/dy2);
		}
		Tnew(0,N-1) += kdt*(ext[0] - 2*Told(0,N-1) + Told(0,N-2))/dx2;
		Tnew(M-1,N-1) += kdt*(ext[M-1] - 2*Told(M-1,N-1) + Told(M-1,N-2))/dx2;
	}
	
	if (boundary_data[0] == NULL){
		double y = 0.0;
		for (int j=0; j < N; j++){
			double x = (jmin + j) * dx;
			Tnew(0, j) = f(x, y);
		}
	}
	if (boundary_data[1] == NULL){
		double y = (imin + M -1) * dy;
		for (int j=0; j < N; j++){
			double x = (jmin + j) * dx;
			Tnew(M-1, j) = f(x, y);
		}
	}
	if (boundary_data[2] == NULL){
		double x = 0.0;
		for (int i=0; i < M; i++){
			double y = (imin + i) * dy;
			Tnew(i, 0) = f(x, y);
		}
	}
	if (boundary_data[3] == NULL){
		double x = (jmin + N - 1) * dx;
		for (int i=0; i < M; i++){
			double y = (imin + i) * dy;
			Tnew(i, N-1) = f(x, y);
		}
	}
}



double f(double x, double y){
	return 1.0;
}

int main(int argc, char* argv[]){
	//Initialize MPI and the communicators
	int poolsize;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &poolsize);

	int dims[2] = {0, 0};
	MPI_Dims_create(poolsize, 2, dims);
	//reorder_dims(dims, nx, ny); //ignored here, useful if nx and ny are very different and the number of processes has few divisors
	int nrow = dims[0];
	int ncol = dims[1];

	int qperiodic[2] = {0, 0};
	MPI_Comm ComArray;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, qperiodic, 1, &ComArray);

	int rank;
	MPI_Comm_rank(ComArray, &rank);
	int coords[2];
	MPI_Cart_coords(ComArray, rank, 2, coords);
	int row = coords[0];
	int col = coords[1];

	int up, down;
	MPI_Cart_shift(ComArray, 0, 1, &up, &down);
	int left, right;
	MPI_Cart_shift(ComArray, 1, 1, &left, &right);
	int neighbours[4] = {up, down, left, right};


	//Physical parameters
	double L = 1.0;
	double H = 1.0;
	double kappa = 1.0;

	for(int nrun=0; nrun < 9; nrun++){
		//simulation parameters
		int nx = 16 * (1 << nrun);
		int ny = 16 * (1 << nrun);

		//Initialize the domain decomposition
		double dx = L/nx;
		double dy = H/ny;
		double dt = 0.5/kappa * (dx*dx*dy*dy)/(dx*dx + dy*dy);
		double kdt = kappa*dt;
		if (rank == 0) printf("dt = 1/%.1f\n", 1/dt);
		double tmax = 2.0/160.0;

		int jmin = (nx*col)/ncol;
		int jmax = (nx*(col+1))/ncol;
		int N = jmax-jmin+1; //local 

		int imin = (ny*row)/nrow;
		int imax = (ny*(row+1))/nrow;
		int M = imax-imin+1;

		double* Told = calloc(sizeof(double), N*M);
		double* Tnew = calloc(sizeof(double), N*M);

		double* boundary_data[4];
		for(int i = 0; i<2; i++){
			boundary_data[i] = neighbours[i] >= 0 ? calloc(sizeof(double), N) : NULL;
			boundary_data[i+2] = neighbours[i+2] >= 0 ? calloc(sizeof(double), M) : NULL;
		}

		//we use this function to set the boundary condition to f
		compute_boundary(Tnew, Told, boundary_data, M, N, f, dx, dy, jmin, imin, kdt);

		MPI_Request send_requests[4];
		MPI_Request recv_requests[4];
		MPI_Datatype ColDtype;
		MPI_Type_vector(M, 1, N, MPI_DOUBLE, &ColDtype);
		MPI_Type_commit(&ColDtype);


		//fill  the boundary arrays before starting
		int maxiter;
		maxiter = (int) ((((double) poolsize)*(1 << 28))/(nx*ny));
		maxiter = maxiter < 100 ? 100 : maxiter;
		maxiter = maxiter > 10000 ? 10000 : maxiter;
		//maxiter = (int) (tmax/dt + 1e-5);
		if (rank == 0) printf("Maxiter: %d, tmax = %f\n", maxiter, maxiter*dt);

		int ti = 0;
		double tsamples[maxiter*6 + 2];
		MPI_Barrier(ComArray);
		tsamples[ti++] = MPI_Wtime();

		i_send(Told, neighbours, M, N, send_requests, ColDtype, ComArray);
		i_recieve(boundary_data, neighbours, M, N, recv_requests, ComArray);
		wait_send(neighbours, send_requests);
		for(int iter = 0; iter < maxiter; iter++){
			wait_recieve(neighbours, recv_requests);
			
			tsamples[ti++] = MPI_Wtime();
			compute_boundary(Told, Tnew, boundary_data, M, N, f, dx, dy, jmin, imin, kdt);
			
			tsamples[ti++] = MPI_Wtime();
			i_recieve(boundary_data, neighbours, M, N, recv_requests, ComArray);
			
			tsamples[ti++] = MPI_Wtime();
			internal_bounds(Told, Tnew, dx, dy, kdt, M, N);
			
			tsamples[ti++] = MPI_Wtime();
			i_send(Tnew, neighbours, M, N, send_requests, ColDtype, ComArray);
			
			tsamples[ti++] = MPI_Wtime();
			compute_inside(Told, Tnew, dx, dy, kdt, M, N);

			tsamples[ti++] = MPI_Wtime();
			wait_send(neighbours, send_requests);
			
			double * temp = Tnew;
			Tnew = Told;
			Told = temp;
			//at the end : Tnew is garbage, Told is good
		}
		wait_recieve(neighbours, recv_requests);
		tsamples[ti++] = MPI_Wtime();

		double tottime = tsamples[maxiter*6+1] - tsamples[0];

		if (rank == 0){
			printf("Time = %fms\n", tottime*1000);
		}
		char fname[200];
		sprintf(fname, "Tsamples/Tsamples-nproc=%d-rank=%d-nx=%d-ny=%d.bin", poolsize, rank, nx, ny);
		dump(tsamples, ti, fname);

		//sprintf(fname, "sol-nx=%d-ny=%d.bin", nx, ny);		
		//dump_to_file(Told, N, M, nx, ny, maxiter, L, H, dt, rank, poolsize, nrow, ncol, ComArray, fname);

		for(int i = 0; i<4; i++){
			if(boundary_data[i] != NULL){
				free(boundary_data[i]);
			}
		}
		free(Told);
		free(Tnew);
		if (rank == 0) printf("\n");

	}


	/*	
	for(int i = 0; i<poolsize; i++){
		MPI_Barrier(ComArray);
		if (rank == i){
			printf("Rank %d of %d : pos(%d,%d), up=%d, down=%d, left=%d, right=%d, imin=%d, imax=%d, jmin=%d, jmax=%d\n", rank, poolsize, row, col, up, down, left, right, imin, imax, jmin,jmax);
			printMatrix(Told, M, N);

			if(up >= 0){
				printf("From above:\n");
				printMatrix(boundary_data[0], 1, N);
			}
			if(down >= 0){
				printf("From below:\n");
				printMatrix(boundary_data[1], 1, N);
			}
			if(left >= 0){
				printf("From left:\n");
				printMatrix(boundary_data[2], M, 1);
			}
			if(right >= 0){
				printf("From right:\n");
				printMatrix(boundary_data[3], M, 1);
			}
		}
		fflush(stdout);
	}
	*/
	


	
	MPI_Finalize();
}

