import torch
import unittest
from gpytorch.gpytorch.utils import batched_tri_to_diag

class TestBatched(unittest.TestCase):
    def test_vectorized_eigenvalues(self):
        t_mat = torch.load('actual_t_mats.pth')
        sym_mat = t_mat.clone()
        b1 = t_mat.size(0)
        b2 = t_mat.size(1)
        b3 = t_mat.size(2)
        alpha = torch.zeros(b1,b2,b3)
        for i in range(b1):
            for j in range(b2):
                alpha[i,j] = t_mat[i,j,:,:].diag()
        alpha.resize_(b1*b2,b3)
        beta = torch.zeros(b1,b2,b3-1)
        for i in range(b1):
            for j in range(b2):
                beta[i,j] = t_mat[i,j,:,:].diag(-1)
        beta.resize_(b1*b2,b3-1)
        val, vec = batched_tridiag_to_diag(alpha, beta)
        sym_val, sym_vec = lanczos_tridiag_to_diag(sym_mat)

        val.resize_(b1,b2,b3)
        res = torch.norm(torch.sort(sym_val)[0] - torch.sort(val)[0])
        self.assertLess(res, 1e-3)

    def test_vectorized_eigenvectors(self):
        t_mat = torch.load('actual_t_mats.pth')
        sym_mat = t_mat.clone()
        b1 = t_mat.size(0)
        b2 = t_mat.size(1)
        b3 = t_mat.size(2)
        alpha = torch.zeros(b1,b2,b3)
        for i in range(b1):
            for j in range(b2):
                alpha[i,j] = t_mat[i,j,:,:].diag()
        alpha.resize_(b1*b2,b3)
        beta = torch.zeros(b1,b2,b3-1)
        for i in range(b1):
            for j in range(b2):
                beta[i,j] = t_mat[i,j,:,:].diag(-1)
        beta.resize_(b1*b2,b3-1)
        val, vec = batched_tridiag_to_diag(alpha, beta)
        sym_val, sym_vec = lanczos_tridiag_to_diag(sym_mat)

        vec.resize_(b1,b2,b3,b3)
        res = torch.norm(torch.sort(torch.abs(sym_vec),-1)[0] - torch.sort(torch.abs(vec),-1)[0])
        self.assertLess(res, 1e-3)

    def test_batched_eigenvalues(self):
        t_mat = torch.load('actual_t_mats.pth')
        sym_mat = t_mat.clone()
        t_val, t_vec = batched_tridiag_to_diag(t_mat)
        sym_val, sym_vec = lanczos_tridiag_to_diag(sym_mat)

        res = torch.norm(torch.sort(sym_val)[0] - torch.sort(t_val)[0])
        self.assertLess(res, 1e-3)

    def test_batched_eigenvectors(self):
        t_mat = torch.load('actual_t_mats.pth')
        sym_mat = t_mat.clone()
        t_val, t_vec = batched_tridiag_to_diag(t_mat)
        sym_val, sym_vec = lanczos_tridiag_to_diag(sym_mat)

        res = torch.norm(torch.sort(torch.abs(sym_vec),-1)[0] - torch.sort(torch.abs(t_vec),-1)[0])
        self.assertLess(res, 1e-3)

if __name__ == '__main__':
    unittest.main()
