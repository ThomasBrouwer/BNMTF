# Run the cross-validation for KBMF

source("cross_val_kbmf.R")
K <- 5
R_values <- c(2,4,6,8,10,12,14,16,18,20)

Px <- 3
Nx <- 622
Pz <- 3
Nz <- 139

# Load in the drug sensitivity values
folder_drug_sensitivity <- '/home/tab43/Dropbox/Biological databases/Sanger_drug_sensivitity/'
name_drug_sensitivity <- 'ic50_excl_empty_filtered_cell_lines_drugs.txt'
Y <- as.matrix(read.table(paste(folder_drug_sensitivity,name_drug_sensitivity,sep=''),
				header=TRUE,
				sep=',',
				colClasses=c(rep("NULL",3), rep("numeric",138))))

# Load in the kernels - X = cancer cell lines, Z = drugs
folder_kernels <- './kernels/'

kernel_copy_variation <- as.matrix(read.table(paste(folder_kernels,'copy_variation.txt',sep=''),header=TRUE))
kernel_gene_expression <- as.matrix(read.table(paste(folder_kernels,'gene_expression.txt',sep=''),header=TRUE))
kernel_mutation <- as.matrix(read.table(paste(folder_kernels,'mutation.txt',sep=''),header=TRUE))

kernel_1d2d <- as.matrix(read.table(paste(folder_kernels,'1d2d_descriptors.txt',sep=''),header=TRUE))
kernel_fingerprints<- as.matrix(read.table(paste(folder_kernels,'PubChem_fingerprints.txt',sep=''),header=TRUE))
kernel_targets <- as.matrix(read.table(paste(folder_kernels,'targets.txt',sep=''),header=TRUE))

Kx <- array(0, c(Nx, Nx, Px))
Kx[,, 1] <- kernel_copy_variation
Kx[,, 2] <- kernel_gene_expression
Kx[,, 3] <- kernel_mutation

Kz <- array(0, c(Nz, Nz, Pz))
Kz[,, 1] <- kernel_1d2d
Kz[,, 2] <- kernel_fingerprints
Kz[,, 3] <- kernel_targets

# Run the cross-validation
kbmf_cross_validation(Kx, Kz, Y, R_values, K)

# Results (5 folds, 200 iterations):
# R:	2		4		6		8		10		12		14		16		18		20
# MSE:  2.832466 	2.448098 	2.294287 	2.227165 	2.243336 	2.259782 	2.283704 	2.309363	2.335845 	2.358715
# R^2:  0.7578040 	0.7906790 	0.8038126 	0.8095175 	0.8081712 	0.8067867 	0.8047146	0.8025545 	0.8002464 	0.7983178
# Rp:   0.8705774 	0.8892419 	0.8965853 	0.8997967 	0.8991491 	0.8985184 	0.8975142	0.8964987 	0.8954419 	0.8944387
