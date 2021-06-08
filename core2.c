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