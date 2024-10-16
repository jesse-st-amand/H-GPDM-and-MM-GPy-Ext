import numpy as np
def circular_mapping_nick(BC):
    '''
    :param BC: backconstraint
    :return: mapping
    '''
    BC.Yt3 = BC.GPNode.X[:, :BC.output_dim - 2]

    BC.Yt3_list = []
    for x0_i, ep_i in zip(BC.GPNode.seq_x0s, BC.GPNode.seq_eps):
        BC.Yt3_list.append(BC.Yt3[x0_i:ep_i + 1, :])

    Yt1_list = []
    Yt2_list = []
    Yt3_list = []

    for i, Yt3_seq in enumerate(BC.Yt3_list):
        Yt3_norm = np.sqrt(Yt3_seq[:, 0] @ Yt3_seq[:, 0] + Yt3_seq[:, 1] @ Yt3_seq[:, 1])
        Yt3_seq[:, :2] = Yt3_seq[:, :2] / Yt3_norm
        angles = np.arctan2(Yt3_seq[:, 1], Yt3_seq[:, 0])

        offStep = np.median(np.abs(angles[1:] - angles[:-1]))

        Yt3_temp = Yt3_seq
        Yt3_temp -= Yt3_temp.mean(0)
        Yt3_temp /= Yt3_temp.std(0)
        Yt3_list.append(Yt3_temp)

        phases = np.linspace(-(np.pi + np.pi / 12) / offStep, (np.pi + np.pi / 12) / offStep,
                             int(BC.GPNode.N / BC.GPNode.num_seqs))[:, None] * offStep

        Yt1_temp = np.cos(phases)
        Yt1_list.append(Yt1_temp)

        Yt2_temp = np.sin(phases)
        Yt2_list.append(Yt2_temp)

    BC.Yt1 = np.vstack(Yt1_list)
    BC.Yt2 = np.vstack(Yt2_list)
    BC.Yt3 = np.vstack(Yt3_list)
    return np.hstack([BC.Yt1, BC.Yt2,BC.Yt3])
    #return BC.GPNode.X.values
def circular_mapping(BC,Y):
    '''
    :param BC: backconstraint
    :return: mapping
    '''
    '''N,D = X.shape
    #BC.Yt3 = X[:, :D - 2]
    BC.Yt3 = X

    BC.Yt3_list = []
    for x0_i, ep_i in zip(BC.GPNode.seq_x0s, BC.GPNode.seq_eps):
        BC.Yt3_list.append(BC.Yt3[x0_i:ep_i + 1, :])'''

    Y_KE = 1 / np.sum((.5 * Y ** 2), 1)



    Yt1_list = []
    Yt2_list = []
    Yt3_list = []

    #Y_KE_list = []

    for i, (x0_i, ep_i) in enumerate(zip(BC.GPNode.seq_x0s, BC.GPNode.seq_eps)):
        #Y_total_dist = np.sqrt(np.sum(np.sum(Y[x0_i:ep_i + 1],0) ** 2))
        #ratio = 2*np.pi / Y_total_dist
        ratio = 2*np.pi /np.sum(Y_KE[x0_i:ep_i + 1])

        #Y_KE_list.append(Y_KE[x0_i:ep_i + 1]*ratio-np.pi)
        #Y_KE_temp
        phases = np.cumsum(Y_KE[x0_i:ep_i + 1]*ratio)
        #offStep = 1
        #phases = np.linspace(-(np.pi + np.pi / 12) / offStep, (np.pi + np.pi / 12) / offStep,
        #                     int(BC.GPNode.N / BC.GPNode.num_seqs))[:, None] * offStep

        Yt1_temp = np.cos(phases)
        Yt1_list.append(Yt1_temp)

        Yt2_temp = np.sin(phases)
        Yt2_list.append(Yt2_temp)

    BC.Yt1 = np.hstack(Yt1_list).reshape([-1,1])
    BC.Yt2 = np.hstack(Yt2_list).reshape([-1,1])
    BC.Yt3 = Y#np.vstack(Yt3_list)
    return BC.Yt1, BC.Yt2,BC.Yt3


def toroidal_mapping(BC,R=5,r=2,n=2):
    """
    R - outer torus radius
    r - inner torus radius
    n - number of winds around torus
    :param BC: backconstraint
    :return: mapping
    """
    BC.R = R
    BC.r = r
    BC.n = n

    BC.Yt4 = BC.GPNode.X[:,:BC.output_dim-3]

    BC.Yt4_list = []
    for x0_i, ep_i in zip(BC.GPNode.seq_x0s, BC.GPNode.seq_eps):
        BC.Yt4_list.append(BC.Yt4[x0_i:ep_i + 1, :])

    Yt1_list = []
    Yt2_list = []
    Yt3_list = []
    Yt4_list = []

    for i, Yt4_seq in enumerate(BC.Yt4_list):
        Yt4_norm = np.sqrt(Yt4_seq[:, 0] @ Yt4_seq[:, 0] + Yt4_seq[:, 1] @ Yt4_seq[:, 1])
        Yt4_seq[:, :2] = Yt4_seq[:, :2] / Yt4_norm
        angles = np.arctan2(Yt4_seq[:, 1], Yt4_seq[:, 0])

        offStep = np.median(np.abs(angles[1:] - angles[:-1]))

        Yt4_temp = Yt4_seq
        Yt4_temp -= Yt4_temp.mean(0)
        Yt4_temp /= Yt4_temp.std(0)
        Yt4_list.append(Yt4_temp)

        # phases = np.linspace(0, (2 * np.pi) / offStep, int(BC.GPNode.N / BC.GPNode.num_seqs))[:, None] * offStep
        phases = np.linspace(-(np.pi + np.pi / 12) / offStep, (np.pi + np.pi / 12) / offStep, int(BC.GPNode.N / BC.GPNode.num_seqs))[:, None] * offStep

        Yt1_temp = (BC.R + BC.r * np.cos(BC.n * phases)) * np.cos(phases)
        Yt1_list.append(Yt1_temp)

        Yt2_temp = (BC.R + BC.r * np.cos(BC.n * phases)) * np.sin(phases)
        Yt2_list.append(Yt2_temp)

        Yt3_temp = BC.r * np.sin(BC.n * phases)
        Yt3_list.append(Yt3_temp)

    BC.Yt1 = np.vstack(Yt1_list)
    BC.Yt2 = np.vstack(Yt2_list)
    BC.Yt3 = np.vstack(Yt3_list)
    BC.Yt4 = np.vstack(Yt4_list)

    return np.hstack([BC.Yt1, BC.Yt2, BC.Yt3, BC.Yt4])

def toroidal_mapping_seqs(BC,Y,R,r,n):
    '''
    R - outer torus radius
    r - inner torus radius
    n - number of winds around torus
    :param BC: backconstraint
    :return: mapping
    '''
    seq_len = BC.GPNode.seq_x0s[1]-BC.GPNode.seq_x0s[0]
    num_seqs = int(Y.shape[0]/seq_len)
    seq_x0s = np.arange(0,Y.shape[0],seq_len)
    seq_eps = np.arange(seq_len-1, Y.shape[0], seq_len)

    Y_KE = 1 / np.sum((.5 * Y ** 2), 1)



    phases_list = []

    for i, (x0_i, ep_i) in enumerate(zip(seq_x0s, seq_eps)):

        ratio = 2*np.pi /np.sum(Y_KE[x0_i:ep_i + 1])
        phases = np.cumsum(Y_KE[x0_i:ep_i + 1]*ratio)
        phases_list.append(phases)

    phases = np.hstack(phases_list).reshape([-1,1])
    Yt1,Yt2,Yt3 = toroid(phases,R,r,n)
    Yt4 = Y
    return Yt1,Yt2,Yt3,Yt4,phases

def toroidal_mapping_pts(BC,Y,R,r,n):
    '''
    R - outer torus radius
    r - inner torus radius
    n - number of winds around torus
    :param BC: backconstraint
    :return: mapping
    '''
    seq_len = BC.GPNode.seq_x0s[1]-BC.GPNode.seq_x0s[0]
    num_seqs = int(Y.shape[0]/seq_len)
    seq_x0s = np.arange(0,Y.shape[0],seq_len)
    seq_eps = np.arange(seq_len-1, Y.shape[0], seq_len)

    Y_KE = 1 / np.sum((.5 * Y ** 2), 1)

    ratio = 2*np.pi /np.sum(Y_KE)
    phases = np.cumsum(Y_KE*ratio)

    Yt1,Yt2,Yt3 = toroid(phases,R,r,n)
    Yt4 = Y
    return Yt1,Yt2,Yt3,Yt4,phases

def toroid(phases,R,r,n):
    return (R + r * np.cos(n * phases)) * np.cos(phases), (R + r * np.cos(n * phases)) * np.sin(phases),r * np.sin(n * phases)

def ellipsoidal_mapping(BC,Y,R,r,n):
    '''
    R - axis 1
    r - axis 2
    n - phase
    :param BC: backconstraint
    :return: mapping
    '''

    Y_KE = 1 / np.sum((.5 * Y ** 2), 1)



    phases_list = []

    for i, (x0_i, ep_i) in enumerate(zip(BC.GPNode.seq_x0s, BC.GPNode.seq_eps)):

        ratio = 2*np.pi /np.sum(Y_KE[x0_i:ep_i + 1])
        phases = np.cumsum(Y_KE[x0_i:ep_i + 1]*ratio)
        phases_list.append(phases)

    phases = np.hstack(phases_list).reshape([-1,1])
    Yt1,Yt2 = ellipse(phases,R,r,n)
    Yt3 = Y
    return Yt1,Yt2,Yt3,phases

def ellipse(phases,R,r,n):
    return R*np.cos(n*phases),r*np.sin(n*phases)