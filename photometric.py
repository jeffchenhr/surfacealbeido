# Import any required libraries here
import cv2                               # OpenCV
import numpy as np                       # numpy
from scipy.optimize import nnls,minimize


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False
    
def lossfunc(solu,G):
    
    B=np.array([[solu[0]**2,0,solu[0]*solu[2]],
              [0,solu[0]**2,solu[0]*solu[3]],
              [0,0,solu[1]**2]])
    
    return np.linalg.norm(G-B)

def photometric(I):
    #First-pass SVD to recover the e and f vector described in paper
    U, D, V = np.linalg.svd(I2,full_matrices=False)

    e=V[0:3,:].T.reshape([w, h, 3])
    f=U[:,0:3]

    print('fsize',f.shape)

    #find the derivative of e
    kernalx=np.array([[0,-0.5,0],
                     [0,0,0],
                     [0,0.5,0]])

    kernaly=np.array([[0,0,0],
                     [-0.5,0,0.5],
                     [0,0,0]])

    e1=e[1:-1,1:-1,0].flatten()
    e2=e[1:-1,1:-1,1].flatten()
    e3=e[1:-1,1:-1,2].flatten()

    e1x=signal.correlate2d(e[:,:,0], kernalx, boundary='fill', mode='valid').flatten()
    e2x=signal.correlate2d(e[:,:,1], kernalx, boundary='fill', mode='valid').flatten()
    e3x=signal.correlate2d(e[:,:,2], kernalx, boundary='fill', mode='valid').flatten()

    e1y=signal.correlate2d(e[:,:,0], kernaly, boundary='fill', mode='valid').flatten()
    e2y=signal.correlate2d(e[:,:,1], kernaly, boundary='fill', mode='valid').flatten()
    e3y=signal.correlate2d(e[:,:,2], kernaly, boundary='fill', mode='valid').flatten()

    #constrcut the integrability contraint matrix

    a11=e2*e3x-e3*e2x
    a21=e2*e3y-e3*e2y
    a13=e1*e2x-e2*e1x
    a23=e1*e2y-e2*e1y
    a12=e1*e3x-e3*e1x
    a22=e1*e3y-e3*e1y

    A=np.concatenate((a11[:,np.newaxis],a12[:,np.newaxis],a13[:,np.newaxis],-a21[:,np.newaxis],-a22[:,np.newaxis],-a23[:,np.newaxis]),axis=1)

    #Find the null space of A which is the solution to Equation Ax=0

    U1,D1,V1=np.linalg.svd(A,full_matrices=False)

    #Reformat solution into 3x2 matrix, which is the first part of the cofactor matrix
    cofactor12=V1[4,:].reshape([3,2],order='F')

    #Assign the remaining value of the co-factor matrix by guessing.
    cofactor=np.concatenate((cofactor12,np.array([0.1,0.01,0.01])[:,np.newaxis]),axis=1)

    #Find inverse of the co-factor matrix
    Pns=np.linalg.lstsq(cofactor, np.eye(3), rcond=None)[0]

    #Rescale the P* to match the cofactor matrix
    Ps=Pns/((Pns[2,1]*Pns[1,2]-Pns[1,1]*Pns[2,2])/cofactor[0,0])

    #Calculate the inverse of P*
    Pstinv=np.linalg.lstsq(Ps.T, np.eye(3), rcond=None)[0]
    #Calcualte fT*f
    ff = np.einsum('ijn,jkn->ikn', f.T.reshape((3,1,I2.shape[0]),order='F'),f[:,:,np.newaxis].T)

    #Construct the unit intensity constraint matrix for b
    ffc=np.concatenate((ff[0,0,:,np.newaxis],2*ff[0,1,:,np.newaxis],2*ff[0,2,:,np.newaxis],ff[1,1,:,np.newaxis],2*ff[1,2,:,np.newaxis],ff[2,2,:,np.newaxis]),axis=1)

    #Solve for QTQ
    Q6= np.linalg.lstsq(ffc, np.ones(I2.shape[0]),rcond=None)[0]

    #Reconstruct QTQ as a symmertric matrix
    QTQ=np.array([[Q6[0],Q6[1],Q6[2]],
                [Q6[1],Q6[3],Q6[4]],
                [Q6[2],Q6[4],Q6[5]]])

    #QTQ is not positive definate due to numerical error introduced by noise, find the closest matrix to QTQ that is positive definate
    QTQ=nearestPD(QTQ)

    #Use cholesky decomposition to find the Qs
    Qs=np.linalg.cholesky(QTQ).T
    
    Qs=np.where(Qs < 0.05, 0, Qs)
    
    #Find the inverse of Qs
    Qsinv=np.linalg.lstsq(Qs, np.eye(3), rcond=None)[0]

    #Compute M
    M=Pstinv.dot(np.diag(D[0:3])).dot(Qsinv)

    x0=np.array([1000,1000,0,0])
    solu=minimize(lossfunc, x0,M.dot(M.T),method="SLSQP",options={'disp': True ,'eps' : 1e0})

    G=np.array([[solu.x[0],0,solu.x[2]],
              [0,solu.x[0],solu.x[3]],
              [0,0,solu.x[1]]])
    
    R=np.linalg.lstsq(G.T, M, rcond=None)[0]
    Q=R.dot(Qs)
    P=G.dot(Ps)

    b_su=np.dot(P,V[0:3,:])
    s_su=np.dot(Q,U[:,0:3].T).T
    s_su=s_su/(np.sqrt(np.sum(np.power(s_su,2),1))[:,np.newaxis])

    p_su = np.sqrt(np.sum(np.power(b_su, 2), 0)).reshape([w, h])
    n_su = b_su.T.reshape([w, h, 3]) / np.tile(np.expand_dims(p_su, 2), [1, 1, 3])

    return p_su,n_su,s_su