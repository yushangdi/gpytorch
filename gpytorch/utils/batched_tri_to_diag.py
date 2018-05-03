import torch

def trid_qr_wshift(t_mat):
    """
    Computes symmetric tridiagonal QR algorithm with implicit Wilkinson shift
    as described in http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf.
    """
    mat = t_mat.clone() # to avoid changing t_mat, actually might not need to clone
    m = mat.size(0) - 1
    err = 10**-8 # Check if this error is correct
    eigenvectors = torch.eye(m+1)
    while (m > 0):
        d = (mat[m-1,m-1] - mat[m,m]) / 2  # Computes Wilkinson's shift
        if d == 0:
            s = mat[m,m] - abs(mat[m,m-1])
        elif d > 0:
            s = mat[m,m] - ((mat[m,m-1])**2 / (d + (d**2 + mat[m,m-1]**2)**0.5))
        elif d < 0:
            s = mat[m,m] - ((mat[m,m-1])**2 / (d - (d**2 + mat[m,m-1]**2)**0.5))
        x = mat[0,0] - s # Implicit QR
        y = mat[1,0]
        for k in range(0,m):
            c = 1
            s = 0
            if abs(y) >= err: # Givens rotation
                if abs(y) > abs(x):
                    tau = -x / y
                    s = (1 + tau**2)**-0.5
                    c = s*tau
                else:
                    tau = -y / x
                    c = (1 + tau**2)**-0.5
                    s = c*tau
            w = c*x - s*y
            d = mat[k,k] - mat[k+1,k+1]
            z = (2*c*mat[k+1,k] + d*s)*s
            mat[k,k] -= z
            mat[k+1,k+1] += z
            mat[k+1,k] = d*c*s + (c*c - s*s)*mat[k+1,k]
            mat[k,k+1] = d*c*s + (c*c - s*s)*mat[k,k+1]
            x = mat[k+1,k]
            if k > 0:
                mat[k,k-1] = w
                mat[k-1,k] = w
            if k < m - 1:
                y = -s*mat[k+2,k+1]
                mat[k+2,k+1] *= c
                mat[k+1,k+2] *= c
            eigenvectors[:,k:k+2] = torch.mm(eigenvectors[:,k:k+2], torch.FloatTensor([[c, s], [-s, c]]))
            print(eigenvectors)
        if abs(mat[m,m-1]) < err*(abs(mat[m-1,m-1]) + abs(mat[m,m])):
            m -= 1
    return torch.diag(mat), eigenvectors

def lanczos_tridiag_to_diag(t_mat):
    """
    Given a num_init_vecs x num_batch x k x k tridiagonal matrix t_mat,
    returns a num_init_vecs x num_batch x k set of eigenvalues
    and a num_init_vecs x num_batch x k x k set of eigenvectors.

    TODO: make the eigenvalue computations done in batch mode.
    """
    t_mat_orig = t_mat
    t_mat = t_mat.cpu()

    if t_mat.dim() == 3:
        t_mat = t_mat.unsqueeze(0)
    b1 = t_mat.size(0)
    b2 = t_mat.size(1)
    n = t_mat.size(2)

    eigenvectors = t_mat.new(*t_mat.shape)
    eigenvalues = t_mat.new(b1, b2, n)

    for i in range(b1):
        for j in range(b2):
            evals, evecs = t_mat[i, j].symeig(eigenvectors=True)
            eigenvectors[i, j] = evecs
            eigenvalues[i, j] = evals

    return eigenvalues.type_as(t_mat_orig), eigenvectors.type_as(t_mat_orig)

def batched_tridiag_to_diag(a, b, err=1e-5, eps=1e-30):
    """
    Given a num_init_vecs*num_batch by k batched vector/matrix alpha
    and a num_init_vecs*num_batch by k-1 batched vector/matrix beta,
    returns a num_init_vecs*num_batch by k set of eigenvalues
    and a num_init_vecs*num_batch by k by k set of eigenvectors.
    Computes symmetric tridiagonal QR algorithm with implicit Wilkinson shift
    as described in http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf.
    """
    a_trsp = a.t().contiguous()
    b_trsp = b.t().contiguous()

    n3, n1n2 = a_trsp.size()

    m = n3 - 1
    eigenvalues = a_trsp.new(n3, n1n2).fill_(0)
    eigenvectors = (
        a_trsp.new(n3).fill_(1).
        diag().
        repeat(n1n2, 1, 1).
        permute(2, 0, 1).contiguous()
    )

    c = a_trsp.new(n1n2)
    s = a_trsp.new(n1n2)
    c2_minus_s2 = a_trsp.new(n1n2)

    d = a_trsp.new(n1n2)
    w = a_trsp.new(n1n2)
    am1 = a_trsp.new(n1n2)
    bm = a_trsp.new(n1n2)

    x = a_trsp.new(n1n2)
    y = a_trsp.new(n1n2)
    z = a_trsp.new(n1n2)
    x2 = a_trsp.new(n1n2)
    y2 = a_trsp.new(n1n2)

    bk_abs = a_trsp.new(n1n2)
    vecs1 = a_trsp.new(n1n2, n3)

    while (m > 0):
        s.copy_(a_trsp[m])
        am1.copy_(a_trsp[m - 1])
        bm.copy_(b_trsp[m - 1]).pow_(2)

        # d = (am1 - s) / 2
        torch.sub(am1, s, out=d).div_(2)  # Computes Wilkinson's shift
        # Come back
        s.sub_(
            bm.div_(
                torch.sign(d.add_(eps)).
                mul_((torch.pow(d, 2).add_(bm)).sqrt_()).
                add_(d)
            )
        )

        # Come back
        x.copy_(s).mul_(-1).add_(a_trsp[0]) # Implicit QR
        y.copy_(b_trsp[0])
        for k in range(0, m):
            torch.pow(x, 2, out=x2)
            torch.pow(y, 2, out=y2)
            torch.add(x2, y2, out=c).rsqrt_()

            s.copy_(c)
            c.mul_(x)
            s.mul_(-y)

            torch.mul(c, x, out=w).addcmul_(-1, s, y)
            torch.sub(a_trsp[k], a_trsp[k + 1], out=d)
            torch.mul(d, s, out=z).addcmul_(2, c, b_trsp[k]).mul_(s)

            # c ** 2 - s ** 2
            torch.pow(c, 2, out=c2_minus_s2)
            c2_minus_s2.addcmul_(-1, s, s)

            # d <- d * c * s
            d.mul_(c).mul_(s)

            a_trsp[k] -= z
            a_trsp[k + 1] += z
            b_trsp[k].mul_(c2_minus_s2).add_(d)

            x.copy_(b_trsp[k])
            if k > 0:
                b_trsp[k - 1].copy_(w)
            if k < m - 1:
                y.copy_(s).mul_(-1).mul_(b_trsp[k + 1])
                b_trsp[k + 1].mul_(c)
            c.unsqueeze_(-1)
            s.unsqueeze_(-1)

            vecs1.copy_(eigenvectors[k])
            eigenvectors[k].mul_(c).addcmul_(-1, eigenvectors[k + 1], s)
            eigenvectors[k + 1].mul_(c).addcmul_(vecs1, s)
            c.squeeze_(-1)
            s.squeeze_(-1)

        torch.abs(b_trsp[k], out=bk_abs)
        if bk_abs.max() < err:
            eigenvalues[m].copy_(a_trsp[m])
            m -= 1

    eigenvalues[0].copy_(a_trsp[0])
    return eigenvalues.t().contiguous(), eigenvectors.permute(1, 2, 0).contiguous()
