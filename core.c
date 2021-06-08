for(int iter = 0; iter < maxiter; iter++){
    wait_recieve(neighbours, recv_requests);
    compute_boundary(Told, Tnew, boundary_data, M, N, f, dx, dy, jmin, imin, kdt);
    i_recieve(boundary_data, neighbours, M, N, recv_requests, ComArray);
    
    internal_bounds(Told, Tnew, dx, dy, kdt, M, N);
    i_send(Tnew, neighbours, M, N, send_requests, ColDtype, ComArray);
    
    compute_inside(Told, Tnew, dx, dy, kdt, M, N);
    
    wait_send(neighbours, send_requests);
    
    double * temp = Tnew;
    Tnew = Told;
    Told = temp;
    //at the end : Tnew is garbage, Told is good
}