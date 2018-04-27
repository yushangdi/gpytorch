import os
import torch
import unittest
from gpytorch.utils.batched_tri_to_diag import batched_tridiag_to_diag, lanczos_tridiag_to_diag


class TestBatched(unittest.TestCase):
    def setUp(self):
        super(TestBatched, self).__init__()
        dirname = os.path.dirname(os.path.realpath(__file__))
        t_mat = torch.load(os.path.join(dirname, 'actual_t_mats.pth'))
        sym_mat = t_mat.clone()
        b1 = t_mat.size(0)
        b2 = t_mat.size(1)
        b3 = t_mat.size(2)
        alpha = torch.zeros(b1, b2, b3)
        for i in range(b1):
            for j in range(b2):
                alpha[i, j] = t_mat[i, j, :, :].diag()
        alpha.resize_(b1 * b2, b3)
        beta = torch.zeros(b1, b2, b3 - 1)
        for i in range(b1):
            for j in range(b2):
                beta[i, j] = t_mat[i, j, :, :].diag(-1)
        beta.resize_(b1 * b2, b3 - 1)

        self.alpha = alpha
        self.beta = beta
        self.sym_mat = sym_mat
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

    def test_vectorized_eigenvalues(self):
        val, vec = batched_tridiag_to_diag(self.alpha.clone(), self.beta.clone())
        sym_val, sym_vec = lanczos_tridiag_to_diag(self.sym_mat.clone())

        val.resize_(self.b1, self.b2, self.b3)
        res = torch.norm(torch.sort(sym_val)[0] - torch.sort(val)[0])
        self.assertLess(res, 1e-3)

    def test_timing(self):
        import time
        n_iter = 20

        alpha = self.alpha.cuda()
        beta = self.beta.cuda()
        sym_mat = self.sym_mat.cuda()
        alpha = alpha + 1 - 1
        beta = beta + 1 - 1
        sym_mat = sym_mat + 1 - 1

        start = time.time()
        orig_alpha = alpha.clone()
        for i in range(n_iter):
            val, vec = batched_tridiag_to_diag(alpha, beta)
        res = time.time() - start
        print('Batched time: %.4f' % (res / n_iter))

        start = time.time()
        for i in range(n_iter):
            sym_val, sym_vec = lanczos_tridiag_to_diag(sym_mat)
        res = time.time() - start
        print('Normal time: %.4f' % (res / n_iter))

    def test_vectorized_eigenvectors(self):
        val, vec = batched_tridiag_to_diag(self.alpha.clone(), self.beta.clone())
        sym_val, sym_vec = lanczos_tridiag_to_diag(self.sym_mat.clone())

        vec.resize_(self.b1, self.b2, self.b3, self.b3)
        res = torch.norm(torch.sort(torch.abs(sym_vec), -1)[0] - torch.sort(torch.abs(vec), -1)[0])
        self.assertLess(res, 1e-3)

if __name__ == '__main__':
    unittest.main()
