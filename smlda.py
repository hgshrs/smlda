#-*- coding: utf-8 -*-

from sklearn.base import BaseEstimator
import numpy as np
import scipy.linalg as la

class UMLDA(BaseEstimator):

    def __init__(self, shape=None, reduced_dim=None, clf=None, constrained_subspaces=[], constrained_dimensions=[], isshown=False, coef_id=0.):
        self.shape = shape
        self.reduced_dim = reduced_dim
        self.isshown = isshown
        self.coef_id = coef_id
        self.clf = clf
        self.constrained_dimensions = constrained_dimensions
        self.n_modes = len(self.shape)

        self.constrained_subspaces = []
        self.constrained_dimensions = []
        for ii in range(self.n_modes):
            self.constrained_subspaces.append([])
            if len(constrained_subspaces) > 0:
                if len(constrained_subspaces[ii]) > 0:
                    self.constrained_subspaces[ii] = constrained_subspaces[ii]
            self.constrained_dimensions.append(0)
            if len(constrained_dimensions) > 0:
                if constrained_dimensions[ii]:
                    if constrained_dimensions[ii] > constrained_subspaces[ii].shape[1]:
                        self.constrained_dimensions[ii] = constrained_subspaces[ii].shape[1]
                    else:
                        self.constrained_dimensions[ii] = constrained_dimensions[ii]

    def fit(self, X, y):
        n_samples = X.shape[0]
        X_ = np.reshape(X, [n_samples] + list(self.shape))

        class_symbols = np.unique(y)
        n_classes = len(class_symbols)

        # constrained subspaces
        constrained_subspaces = []
        for ii in range(self.n_modes):
            constrained_subspaces.append([])
            if self.constrained_dimensions:
                if self.constrained_dimensions[ii]:
                    if self.constrained_dimensions[ii] > self.constrained_subspaces[ii].shape[1]:
                        constrained_subspaces[ii] = self.constrained_subspaces[ii]
                    else:
                        constrained_subspaces[ii] = self.constrained_subspaces[ii][:, :self.constrained_dimensions[ii]]

        # centering
        M0 = X_.mean(0)
        Xb = np.empty([n_classes]+self.shape)
        for ii in range(len(class_symbols)):
            idxs = np.where(y == class_symbols[ii])[0]
            Xb[ii] = np.sqrt(idxs.shape[0])*(X_[idxs].mean(0) - M0)
        Xw = np.empty([n_samples]+self.shape)
        for ii in range(n_samples):
            Xw[ii] = X_[ii] - X_[y==y[ii]].mean(0)


        reduced_shape = np.ones(self.n_modes, dtype=int)
        self.fit_costs = []
        self.coefs = []
        G = []
        if np.array(self.constrained_dimensions).min() > 0:
            if self.reduced_dim > np.array(self.constrained_dimensions).min():
                self.reduced_dim = np.array(self.constrained_dimensions).min()
        Yp = np.zeros((n_samples, self.reduced_dim), dtype=float)
        for ii in xrange(self.reduced_dim):
            U, costs = ray2(Xb, Xw, np.hstack((0, reduced_shape)), S=[1] + constrained_subspaces,
                    G=G, isshown=self.isshown, coef_id=self.coef_id)
            self.fit_costs.append(costs)
            self.coefs.append(U)

            yp = self.emp_transform(X, [self.coefs[-1]])
            yp /= la.norm(yp)
            G.append(mprod(X_, yp, 0))
            Yp[:, ii] = yp.ravel()

        if self.clf:
            A = self.emp_transform(X, self.coefs)
            self.clf.fit(A, y)

        return self

    def transform(self, X):
        A = self.emp_transform(X, self.coefs)
        if self.clf:
            return self.clf.transform(A)
        else:
            return A

    def multilinear_transform(self, X, U):
        if X.shape[1] == 1:
            X = X[np.newaxis, :]
        n_samples = X.shape[0]
        X_ = np.reshape(X, [n_samples] + list(self.shape))
        return multilinear_transform(X_, U)

    def emp_transform(self, X, coefs):
        reduced_dim = len(coefs)
        A = []
        for ii in xrange(reduced_dim):
            A.append(self.multilinear_transform(X, coefs[ii]).ravel())
        A = np.array(A).T
        return A
    
    def predict(self, X):
        A = self.emp_transform(X, self.coefs)
        return self.clf.predict(A)

    def score(self, X, y):
        A = self.emp_transform(X, self.coefs)
        return self.clf.score(A, y)

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, **params):
        return {'shape':self.shape, 'reduced_dim':self.reduced_dim, 'isshown':self.isshown,
                'coef_id':self.coef_id, 'clf':self.clf, 'constrained_subspaces':self.constrained_subspaces,
                'constrained_dimensions':self.constrained_dimensions}


def ray2(X1, X2, rrank, S=[], G=[], max_iter=1000, min_diff=1e-6, isshown=False, coef_id=0):
    dim = np.array(X1.shape)
    n_dim = len(dim)
    rdim = np.array(np.where(rrank > 0)[0])
    scost = np.inf
    costpre = 0
    A = []
    for ii in range(n_dim):
        A.append([])
        S.append([])
    uncorr_ = False
    if len(G) > 0:
        uncorr_ = True
    n_iter = 0
    cost_iter = np.zeros((max_iter+1, 1), dtype='float64')
    while scost > min_diff:
        for ii in rdim:

            A[ii] = []
            X1_trans = multilinear_transform(X1, A)
            X2_trans = multilinear_transform(X2, A)

            R1 = tensor_cov_mode(X1_trans, ii)
            if len(X2) == 0:
                R2 = np.eye(R1.shape[0])
            else:
                R2 = tensor_cov_mode(X2_trans, ii)
            R2 += coef_id*np.eye(R2.shape[0])
            Q1 = R1[:]
            Q2 = R2[:]
            Sn = S[ii][:]

            # uncorrelate
            if uncorr_ and (n_iter > 0):
                UC = []
                for jj in xrange(len(G)):
                    tmp = multilinear_transform(G[jj], A).squeeze()
                    UC.append(tmp)
                UC = np.array(UC).T

                if len(Sn) > 0:
                    US = spaninter(Sn, null(UC))
                else:
                    US = null(UC)
                Sn = US

            if len(Sn) > 0:
                Q1 = quadcalc(Q1, Sn)
                if len(X2) > 0:
                    Q2 = quadcalc(Q2, Sn)
                else:
                    Q2 = np.eye(Q1.shape[0])
            else:
                Q1 = R1
                if len(X2) > 0:
                    Q2 = Q2
                else:
                    Q2 = np.eye(Q1.shape[0])

            l, v = la.eigh(Q1, Q2)
            idx = l.argsort()[::-1]
            l = l[idx]
            v = v[:, idx]

            if len(Sn) > 0:
                A[ii] = np.dot(Sn, v[:, :rrank[ii]])
            else:
                A[ii] = v[:, :rrank[ii]]
            if ii == 1:
                A[ii] = A[ii]/la.norm(A[ii])

        out1 = quadcalc(R1, A[rdim[-1]])
        out2 = quadcalc(R2, A[rdim[-1]])
        cost = np.trace(out1)/np.trace(out2)
        scost = cost - costpre
        scost = np.abs(scost)
        costpre = cost
        if n_iter < 1:
            scost = np.inf
            cost = np.NaN
        if isshown:
            print '%d: %.4e (%.4e)\r' % (n_iter, cost, scost),
        cost_iter[n_iter, 0] = cost
        n_iter += 1
        if n_iter > max_iter:
            break
    cost_iter = cost_iter[:n_iter, 0]
    if isshown:
        print ''

    return A, cost_iter

def multilinear_transform(X, A):
    sz = X.shape
    for jj in range(len(sz)):
        if len(A[jj]) > 0:
            X = mprod(X, A[jj], jj)
    return X

def tensor_cov_mode(t, mode, N=1, subtract_mean=0):
    t_unfold = unfold(t, mode)
    if subtract_mean == 1:
        t_unfold = subtract_mean_mode(t_unfold, 1)
    out = np.dot(t_unfold, t_unfold.T)/N
    return out

def null(X):
    M = np.dot(X, X.T)
    l, v = la.eigh(M)
    N = v[:, np.where(l < 1e-6)[0]]
    return N

def spaninter(X, Y):
    Ic = la.orth(np.hstack([null(X), null(Y)]))
    S = null(Ic)
    return S

# A'*R*A
def quadcalc(R, A):
    return np.dot(np.dot(A.T, R), A)

def mprod(t, x, mode):
    dim = np.array(t.shape)
    t_unfold = unfold(t, mode)
    ans = np.dot(x.T, t_unfold)
    dim[mode] = x.shape[1];
    out = fold(ans, mode, dim)
    return out

def unfold(t, mode):
    dim = np.array(t.shape)
    if mode == 0:
        v2 = range(mode+1, len(dim))
        out = t.reshape(dim[0], np.prod(dim[v2]))
    else:
        v2 = range(len(dim))
        v2[0] = mode
        v2[mode] = 0
        outt = t.swapaxes(0, mode)
        out = outt.reshape(dim[mode], np.prod(dim[v2[1:len(dim)]]))
    return out

def subtract_mean_mode(t, mode):
    dim = np.array(t.shape)
    tdim = dim.copy()
    tdim[mode] = 1
    m = t.mean(mode)
    m.resize(tdim)
    mm = mprod(m, np.ones((1, dim[mode]), dtype='float64'), mode)
    out = t - mm
    return out

def fold(mat, mode, dim):
    if mode == 0:
        out = mat.reshape(dim)
    else:
        v3 = range(len(dim))
        v3[0] = mode
        v3[mode] = 0
        tmp = mat.reshape(dim[v3])
        out = tmp.swapaxes(0, mode)
    return out

if __name__ == '__main__':
    from sklearn import cross_validation
    from sklearn.lda import LDA

    print 'Demo code of SMLDA with random samples'

    # set parameters
    n_samples = 50
    signal_shape = [10]*3
    n_modes = len(signal_shape)
    sn_coef = 5.
    n_test_samples = 10
    n_iter_cv = 5
    reduced_dim = np.array(signal_shape).min()/2
    print 'Tensor size: %s' % signal_shape
    print '\t--> Reduced to: [%s]' % reduced_dim

    # make samples
    S1 = np.random.standard_normal(signal_shape)
    S2 = np.random.standard_normal(signal_shape)
    X_shaped = np.concatenate([
        np.tile(S1[np.newaxis], [n_samples/2] + [1]*n_modes),
        np.tile(S2[np.newaxis], [n_samples/2] + [1]*n_modes),
        ], 0)
    X_shaped += sn_coef * np.random.standard_normal([n_samples]+signal_shape)
    X = np.reshape(X_shaped, [n_samples, np.prod(signal_shape)])
    y = np.hstack([
        np.ones(n_samples/2)*1,
        np.ones(n_samples/2)*-1,
        ])
    shuffle_order = np.random.permutation(n_samples)
    X = X[shuffle_order]
    y = y[shuffle_order]

    # cross validation with LDA
    cv = cross_validation.StratifiedShuffleSplit(y, n_iter=n_iter_cv, test_size=n_test_samples, random_state=0)
    lda = LDA()
    scores = cross_validation.cross_val_score(lda, X, y, cv=cv, n_jobs=1, verbose=0)
    print 'CV scores:'
    print '\tLDA:\t%.2f%%' % (scores.mean()*100)

    # cross validation with UMLDA
    clf = UMLDA(shape=signal_shape, reduced_dim=reduced_dim, isshown=False, coef_id=0, clf=lda)
    scores = cross_validation.cross_val_score(clf, X, y, cv=cv, n_jobs=1, verbose=0)
    print '\tUMLDA:\t%.2f%%' % (scores.mean()*100)

    # generate random orthogonal basis for UMLDA
    subspaces = []
    for mode_idx in range(n_modes):
        randmat = np.random.randn(signal_shape[mode_idx], signal_shape[mode_idx])
        subspaces += [la.orth(randmat)]

    # cross validation with SMLDA
    clf = UMLDA(shape=signal_shape, reduced_dim=reduced_dim,
            constrained_subspaces=subspaces, constrained_dimensions=np.array(signal_shape)/2,
            isshown=False, coef_id=0, clf=lda)
    scores = cross_validation.cross_val_score(clf, X, y, cv=cv, n_jobs=1, verbose=0)
    print '\tSMLDA:\t%.2f%%' % (scores.mean()*100)
