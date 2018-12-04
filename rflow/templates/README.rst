
===========================
Template Submission Scripts
===========================

Note that we do not want to use the node exclusively,
especially when using nodes with multiple GPUs (on lobos: k40 -- 2 GPUs, pascal -- 4 GPUs).
OpenMM does all the work on the GPU and usually utilizes only one GPU per simulation.
By requesting only on GPU per job, the rest of the GPUs can be utilized by other jobs.
