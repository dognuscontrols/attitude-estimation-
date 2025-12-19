This repo implements a two-stage pipeline for rigid-body attitude:

1.	Attitude estimation (Wahba’s problem): solve a convex lift of Wahba’s trace maximization using a 4x4 PSD, trace-one SDP (SCS), and recover an attitude estimate via the principal eigenvector of an associated 4x4 matrix.
   
3.	Attitude regulation: track the estimated target with a bounded-torque convex MPC controller posed as a QP (OSQP) on a small-angle linearization in intrinsic error coordinates.
