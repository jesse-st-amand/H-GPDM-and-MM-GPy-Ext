import numpy as np
import matplotlib
import matplotlib.animation as animation
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

class InteractiveFigures():
    def __init__(self,list_of_stick_dicts):
        self.list_of_stick_dicts = list_of_stick_dicts
        self.num_figures = len(list_of_stick_dicts)

    def plot_animation_all_figures(self):
        x_dims = [-20, 20]
        y_dims = [-30, 5]
        z_dims = [0, 25]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim3d([x_dims[0],x_dims[1]])
        ax.set_xlabel('X')

        ax.set_ylim3d([y_dims[0],y_dims[1]])
        ax.set_ylabel('Y')

        ax.set_zlim3d([z_dims[0],z_dims[1]])
        ax.set_zlabel('Z')

        ax.set_title('3D Animated Skeleton')

        # Provide starting angle for the view.
        ax.view_init(10, 10)

        colors = ['b','r','g','y','m','c']
        colors = ['b', 'r', 'g', 'y', 'm', 'c','b', 'r', 'g', 'y', 'm', 'c','b', 'r', 'g', 'y', 'm', 'c','b', 'r', 'g', 'y', 'm', 'c','b', 'r', 'g', 'y', 'm', 'c','b', 'r', 'g', 'y', 'm', 'c']
        lws = [6,5.5,5,4.5,4,3.5]
        lws = [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]
        markersizes = [20,19,18,17,16,15]
        markersizes = [20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
        line = '-'
        num_figures = self.num_figures
        list_of_stick_dicts = self.list_of_stick_dicts

        bodies = []

        larm_middles = []
        lthumbs = []
        lindices = []
        lrings = []
        lpinkies = []

        rarm_middles = []
        rthumbs = []
        rindices = []
        rrings = []
        rpinkies = []

        for i,stick_dicts in enumerate(list_of_stick_dicts):
            print(i)
            lw = lws[i]
            color = colors[i]
            markersize = markersizes[i]

            bodies.append(ax.plot3D(np.array([]),np.array([]),np.array([]), line, lw=lw, color=color))

            larm_middles.append(ax.plot3D(np.array([]),np.array([]),np.array([]), line, lw=lw, color=color))
            lthumbs.append(ax.plot3D(np.array([]),np.array([]),np.array([]), line, lw=lw, color=color))
            lindices.append(ax.plot3D(np.array([]),np.array([]),np.array([]), line, lw=lw, color=color))
            lrings.append(ax.plot3D(np.array([]),np.array([]),np.array([]), line, lw=lw, color=color))
            lpinkies.append(ax.plot3D(np.array([]),np.array([]),np.array([]), line, lw=lw, color=color))

            rarm_middles.append(ax.plot3D(np.array([]),np.array([]),np.array([]), line, lw=lw, color=color))
            rthumbs.append(ax.plot3D(np.array([]),np.array([]),np.array([]), line, lw=lw, color=color))
            rindices.append(ax.plot3D(np.array([]),np.array([]),np.array([]), line, lw=lw, color=color))
            rrings.append(ax.plot3D(np.array([]),np.array([]),np.array([]), line, lw=lw, color=color))
            rpinkies.append(ax.plot3D(np.array([]),np.array([]),np.array([]), line, lw=lw, color=color))

        def animate(t):
            for i,stick_dicts in enumerate(list_of_stick_dicts):
                bodyx = np.array([stick_dicts["Head_End"]["CC"][t, 0], stick_dicts["Head"]["CC"][t, 0],
                                        stick_dicts["Spine"]["CC"][t, 0], stick_dicts["Hips"]["CC"][t, 0]])
                bodyy = np.array([stick_dicts["Head_End"]["CC"][t, 1], stick_dicts["Head"]["CC"][t, 1],
                                        stick_dicts["Spine"]["CC"][t, 1], stick_dicts["Hips"]["CC"][t, 1]])
                bodyz = np.array([stick_dicts["Head_End"]["CC"][t, 2], stick_dicts["Head"]["CC"][t, 2],
                                        stick_dicts["Spine"]["CC"][t, 2], stick_dicts["Hips"]["CC"][t, 2]])
                bodies
                exec('body' + str(i) + '= bodies[' + str(i) + '][0]')
                exec('body' + str(i) + '.set_data(bodyx, bodyy)')
                exec('body' + str(i) + '.set_3d_properties(bodyz)')

                larm_middlex = np.array(
                    [stick_dicts["LeftShoulder"]["CC"][t, 0], stick_dicts["LeftArm"]["CC"][t, 0],
                     stick_dicts["LeftForeArm"]["CC"][t, 0], stick_dicts["LeftHand"]["CC"][t, 0],
                     stick_dicts["LeftHandMiddle1"]["CC"][t, 0], stick_dicts["LeftHandMiddle2"]["CC"][t, 0],
                     stick_dicts["LeftHandMiddle3"]["CC"][t, 0], stick_dicts["LeftHandMiddle3_End"]["CC"][t, 0]])
                larm_middley = np.array(
                    [stick_dicts["LeftShoulder"]["CC"][t, 1], stick_dicts["LeftArm"]["CC"][t, 1],
                     stick_dicts["LeftForeArm"]["CC"][t, 1], stick_dicts["LeftHand"]["CC"][t, 1],
                     stick_dicts["LeftHandMiddle1"]["CC"][t, 1], stick_dicts["LeftHandMiddle2"]["CC"][t, 1],
                     stick_dicts["LeftHandMiddle3"]["CC"][t, 1], stick_dicts["LeftHandMiddle3_End"]["CC"][t, 1]])
                larm_middlez = np.array(
                    [stick_dicts["LeftShoulder"]["CC"][t, 2], stick_dicts["LeftArm"]["CC"][t, 2],
                     stick_dicts["LeftForeArm"]["CC"][t, 2], stick_dicts["LeftHand"]["CC"][t, 2],
                     stick_dicts["LeftHandMiddle1"]["CC"][t, 2], stick_dicts["LeftHandMiddle2"]["CC"][t, 2],
                     stick_dicts["LeftHandMiddle3"]["CC"][t, 2], stick_dicts["LeftHandMiddle3_End"]["CC"][t, 2]])
                larm_middles
                exec('larm_middle' + str(i) + '= larm_middles[' + str(i) + '][0]')
                exec('larm_middle' + str(i) + '.set_data(larm_middlex, larm_middley)')
                exec('larm_middle' + str(i) + '.set_3d_properties(larm_middlez)')

                lindexx = np.array(
                    [stick_dicts["LeftHand"]["CC"][t, 0], stick_dicts["LeftHandIndex"]["CC"][t, 0],
                     stick_dicts["LeftHandIndex1"]["CC"][t, 0], stick_dicts["LeftHandIndex2"]["CC"][t, 0],
                     stick_dicts["LeftHandIndex3"]["CC"][t, 0], stick_dicts["LeftHandIndex3_End"]["CC"][t, 0]])
                lindexy = np.array(
                    [stick_dicts["LeftHand"]["CC"][t, 1], stick_dicts["LeftHandIndex"]["CC"][t, 1],
                     stick_dicts["LeftHandIndex1"]["CC"][t, 1], stick_dicts["LeftHandIndex2"]["CC"][t, 1],
                     stick_dicts["LeftHandIndex3"]["CC"][t, 1], stick_dicts["LeftHandIndex3_End"]["CC"][t, 1]])
                lindexz = np.array(
                    [stick_dicts["LeftHand"]["CC"][t, 2], stick_dicts["LeftHandIndex"]["CC"][t, 2],
                     stick_dicts["LeftHandIndex1"]["CC"][t, 2], stick_dicts["LeftHandIndex2"]["CC"][t, 2],
                     stick_dicts["LeftHandIndex3"]["CC"][t, 2], stick_dicts["LeftHandIndex3_End"]["CC"][t, 2]])
                lindices
                exec('lindex' + str(i) + '= lindices[' + str(i) + '][0]')
                exec('lindex' + str(i) + '.set_data(lindexx, lindexy)')
                exec('lindex' + str(i) + '.set_3d_properties(lindexz)')

                lthumbx = np.array(
                    [stick_dicts["LeftHandIndex"]["CC"][t, 0], stick_dicts["LeftHandThumb1"]["CC"][t, 0],
                     stick_dicts["LeftHandThumb2"]["CC"][t, 0], stick_dicts["LeftHandThumb3"]["CC"][t, 0], stick_dicts["LeftHandThumb3_End"]["CC"][t, 0]])
                lthumby = np.array(
                    [stick_dicts["LeftHandIndex"]["CC"][t, 1], stick_dicts["LeftHandThumb1"]["CC"][t, 1],
                     stick_dicts["LeftHandThumb2"]["CC"][t, 1], stick_dicts["LeftHandThumb3"]["CC"][t, 1], stick_dicts["LeftHandThumb3_End"]["CC"][t, 1]])
                lthumbz = np.array(
                    [stick_dicts["LeftHandIndex"]["CC"][t, 2], stick_dicts["LeftHandThumb1"]["CC"][t, 2],
                     stick_dicts["LeftHandThumb2"]["CC"][t, 2], stick_dicts["LeftHandThumb3"]["CC"][t, 2], stick_dicts["LeftHandThumb3_End"]["CC"][t, 2]])
                lthumbs
                exec('lthumb' + str(i) + '= lthumbs[' + str(i) + '][0]')
                exec('lthumb' + str(i) + '.set_data(lthumbx, lthumby)')
                exec('lthumb' + str(i) + '.set_3d_properties(lthumbz)')

                lringx = np.array(
                    [stick_dicts["LeftHand"]["CC"][t, 0], stick_dicts["LeftHandRing"]["CC"][t, 0],
                     stick_dicts["LeftHandRing1"]["CC"][t, 0], stick_dicts["LeftHandRing2"]["CC"][t, 0],
                     stick_dicts["LeftHandRing3"]["CC"][t, 0], stick_dicts["LeftHandRing3_End"]["CC"][t, 0]])
                lringy = np.array(
                    [stick_dicts["LeftHand"]["CC"][t, 1], stick_dicts["LeftHandRing"]["CC"][t, 1],
                     stick_dicts["LeftHandRing1"]["CC"][t, 1], stick_dicts["LeftHandRing2"]["CC"][t, 1],
                     stick_dicts["LeftHandRing3"]["CC"][t, 1], stick_dicts["LeftHandRing3_End"]["CC"][t, 1]])
                lringz = np.array(
                    [stick_dicts["LeftHand"]["CC"][t, 2], stick_dicts["LeftHandRing"]["CC"][t, 2],
                     stick_dicts["LeftHandRing1"]["CC"][t, 2], stick_dicts["LeftHandRing2"]["CC"][t, 2],
                     stick_dicts["LeftHandRing3"]["CC"][t, 2], stick_dicts["LeftHandRing3_End"]["CC"][t, 2]])
                lrings
                exec('lring' + str(i) + '= lrings[' + str(i) + '][0]')
                exec('lring' + str(i) + '.set_data(lringx, lringy)')
                exec('lring' + str(i) + '.set_3d_properties(lringz)')

                lpinkyx = np.array(
                    [stick_dicts["LeftHandRing"]["CC"][t, 0], stick_dicts["LeftHandPinky"]["CC"][t, 0],
                     stick_dicts["LeftHandPinky1"]["CC"][t, 0], stick_dicts["LeftHandPinky2"]["CC"][t, 0],
                     stick_dicts["LeftHandPinky3"]["CC"][t, 0], stick_dicts["LeftHandPinky3_End"]["CC"][t, 0]])
                lpinkyy = np.array(
                    [stick_dicts["LeftHandRing"]["CC"][t, 1], stick_dicts["LeftHandPinky"]["CC"][t, 1],
                     stick_dicts["LeftHandPinky1"]["CC"][t, 1], stick_dicts["LeftHandPinky2"]["CC"][t, 1],
                     stick_dicts["LeftHandPinky3"]["CC"][t, 1], stick_dicts["LeftHandPinky3_End"]["CC"][t, 1]])
                lpinkyz = np.array(
                    [stick_dicts["LeftHandRing"]["CC"][t, 2], stick_dicts["LeftHandPinky"]["CC"][t, 2],
                     stick_dicts["LeftHandPinky1"]["CC"][t, 2], stick_dicts["LeftHandPinky2"]["CC"][t, 2],
                     stick_dicts["LeftHandPinky3"]["CC"][t, 2], stick_dicts["LeftHandPinky3_End"]["CC"][t, 2]])
                lpinkies
                exec('lpinky' + str(i) + '= lpinkies[' + str(i) + '][0]')
                exec('lpinky' + str(i) + '.set_data(lpinkyx, lpinkyy)')
                exec('lpinky' + str(i) + '.set_3d_properties(lpinkyz)')

                rarm_middlex = np.array(
                    [stick_dicts["RightShoulder"]["CC"][t, 0], stick_dicts["RightArm"]["CC"][t, 0],
                     stick_dicts["RightForeArm"]["CC"][t, 0], stick_dicts["RightHand"]["CC"][t, 0],
                     stick_dicts["RightHandMiddle1"]["CC"][t, 0], stick_dicts["RightHandMiddle2"]["CC"][t, 0],
                     stick_dicts["RightHandMiddle3"]["CC"][t, 0], stick_dicts["RightHandMiddle3_End"]["CC"][t, 0]])
                rarm_middley = np.array(
                    [stick_dicts["RightShoulder"]["CC"][t, 1], stick_dicts["RightArm"]["CC"][t, 1],
                     stick_dicts["RightForeArm"]["CC"][t, 1], stick_dicts["RightHand"]["CC"][t, 1],
                     stick_dicts["RightHandMiddle1"]["CC"][t, 1], stick_dicts["RightHandMiddle2"]["CC"][t, 1],
                     stick_dicts["RightHandMiddle3"]["CC"][t, 1], stick_dicts["RightHandMiddle3_End"]["CC"][t, 1]])
                rarm_middlez = np.array(
                    [stick_dicts["RightShoulder"]["CC"][t, 2], stick_dicts["RightArm"]["CC"][t, 2],
                     stick_dicts["RightForeArm"]["CC"][t, 2], stick_dicts["RightHand"]["CC"][t, 2],
                     stick_dicts["RightHandMiddle1"]["CC"][t, 2], stick_dicts["RightHandMiddle2"]["CC"][t, 2],
                     stick_dicts["RightHandMiddle3"]["CC"][t, 2], stick_dicts["RightHandMiddle3_End"]["CC"][t, 2]])
                rarm_middles
                exec('rarm_middle' + str(i) + '= rarm_middles[' + str(i) + '][0]')
                exec('rarm_middle' + str(i) + '.set_data(rarm_middlex, rarm_middley)')
                exec('rarm_middle' + str(i) + '.set_3d_properties(rarm_middlez)')

                rindexx = np.array(
                    [stick_dicts["RightHand"]["CC"][t, 0], stick_dicts["RightHandIndex"]["CC"][t, 0],
                     stick_dicts["RightHandIndex1"]["CC"][t, 0], stick_dicts["RightHandIndex2"]["CC"][t, 0],
                     stick_dicts["RightHandIndex3"]["CC"][t, 0], stick_dicts["RightHandIndex3_End"]["CC"][t, 0]])
                rindexy = np.array(
                    [stick_dicts["RightHand"]["CC"][t, 1], stick_dicts["RightHandIndex"]["CC"][t, 1],
                     stick_dicts["RightHandIndex1"]["CC"][t, 1], stick_dicts["RightHandIndex2"]["CC"][t, 1],
                     stick_dicts["RightHandIndex3"]["CC"][t, 1], stick_dicts["RightHandIndex3_End"]["CC"][t, 1]])
                rindexz = np.array(
                    [stick_dicts["RightHand"]["CC"][t, 2], stick_dicts["RightHandIndex"]["CC"][t, 2],
                     stick_dicts["RightHandIndex1"]["CC"][t, 2], stick_dicts["RightHandIndex2"]["CC"][t, 2],
                     stick_dicts["RightHandIndex3"]["CC"][t, 2], stick_dicts["RightHandIndex3_End"]["CC"][t, 2]])
                rindices
                exec('rindex' + str(i) + '= rindices[' + str(i) + '][0]')
                exec('rindex' + str(i) + '.set_data(rindexx, rindexy)')
                exec('rindex' + str(i) + '.set_3d_properties(rindexz)')

                rthumbx = np.array(
                    [stick_dicts["RightHandIndex"]["CC"][t, 0], stick_dicts["RightHandThumb1"]["CC"][t, 0],
                     stick_dicts["RightHandThumb2"]["CC"][t, 0], stick_dicts["RightHandThumb3"]["CC"][t, 0],
                     stick_dicts["RightHandThumb3_End"]["CC"][t, 0]])
                rthumby = np.array(
                    [stick_dicts["RightHandIndex"]["CC"][t, 1], stick_dicts["RightHandThumb1"]["CC"][t, 1],
                     stick_dicts["RightHandThumb2"]["CC"][t, 1], stick_dicts["RightHandThumb3"]["CC"][t, 1],
                     stick_dicts["RightHandThumb3_End"]["CC"][t, 1]])
                rthumbz = np.array(
                    [stick_dicts["RightHandIndex"]["CC"][t, 2], stick_dicts["RightHandThumb1"]["CC"][t, 2],
                     stick_dicts["RightHandThumb2"]["CC"][t, 2], stick_dicts["RightHandThumb3"]["CC"][t, 2],
                     stick_dicts["RightHandThumb3_End"]["CC"][t, 2]])
                rthumbs
                exec('rthumb' + str(i) + '= rthumbs[' + str(i) + '][0]')
                exec('rthumb' + str(i) + '.set_data(rthumbx, rthumby)')
                exec('rthumb' + str(i) + '.set_3d_properties(rthumbz)')

                rringx = np.array(
                    [stick_dicts["RightHand"]["CC"][t, 0], stick_dicts["RightHandRing"]["CC"][t, 0],
                     stick_dicts["RightHandRing1"]["CC"][t, 0], stick_dicts["RightHandRing2"]["CC"][t, 0],
                     stick_dicts["RightHandRing3"]["CC"][t, 0], stick_dicts["RightHandRing3_End"]["CC"][t, 0]])
                rringy = np.array(
                    [stick_dicts["RightHand"]["CC"][t, 1], stick_dicts["RightHandRing"]["CC"][t, 1],
                     stick_dicts["RightHandRing1"]["CC"][t, 1], stick_dicts["RightHandRing2"]["CC"][t, 1],
                     stick_dicts["RightHandRing3"]["CC"][t, 1], stick_dicts["RightHandRing3_End"]["CC"][t, 1]])
                rringz = np.array(
                    [stick_dicts["RightHand"]["CC"][t, 2], stick_dicts["RightHandRing"]["CC"][t, 2],
                     stick_dicts["RightHandRing1"]["CC"][t, 2], stick_dicts["RightHandRing2"]["CC"][t, 2],
                     stick_dicts["RightHandRing3"]["CC"][t, 2], stick_dicts["RightHandRing3_End"]["CC"][t, 2]])
                rrings
                exec('rring' + str(i) + '= rrings[' + str(i) + '][0]')
                exec('rring' + str(i) + '.set_data(rringx, rringy)')
                exec('rring' + str(i) + '.set_3d_properties(rringz)')

                rpinkyx = np.array(
                    [stick_dicts["RightHandRing"]["CC"][t, 0], stick_dicts["RightHandPinky"]["CC"][t, 0],
                     stick_dicts["RightHandPinky1"]["CC"][t, 0], stick_dicts["RightHandPinky2"]["CC"][t, 0],
                     stick_dicts["RightHandPinky3"]["CC"][t, 0], stick_dicts["RightHandPinky3_End"]["CC"][t, 0]])
                rpinkyy = np.array(
                    [stick_dicts["RightHandRing"]["CC"][t, 1], stick_dicts["RightHandPinky"]["CC"][t, 1],
                     stick_dicts["RightHandPinky1"]["CC"][t, 1], stick_dicts["RightHandPinky2"]["CC"][t, 1],
                     stick_dicts["RightHandPinky3"]["CC"][t, 1], stick_dicts["RightHandPinky3_End"]["CC"][t, 1]])
                rpinkyz = np.array(
                    [stick_dicts["RightHandRing"]["CC"][t, 2], stick_dicts["RightHandPinky"]["CC"][t, 2],
                     stick_dicts["RightHandPinky1"]["CC"][t, 2], stick_dicts["RightHandPinky2"]["CC"][t, 2],
                     stick_dicts["RightHandPinky3"]["CC"][t, 2], stick_dicts["RightHandPinky3_End"]["CC"][t, 2]])
                rpinkies
                exec('rpinky' + str(i) + '= rpinkies[' + str(i) + '][0]')
                exec('rpinky' + str(i) + '.set_data(rpinkyx, rpinkyy)')
                exec('rpinky' + str(i) + '.set_3d_properties(rpinkyz)')

            return exec(','.join(['body'+str(i)+', larm_middle'+str(i)+', lthumb'+str(i)+', lpinky'+str(i)+', lindex'+str(i)+', lring'+str(i)+', rarm_middle'+str(i)+', rthumb'+str(i)+', rpinky'+str(i)+', rindex'+str(i)+', rring'+str(i) for i in range(num_figures)]))

        return fig, animate


class InteractiveFiguresCMU():
    def __init__(self,list_of_stick_dicts):
        self.list_of_stick_dicts = list_of_stick_dicts
        self.num_figures = len(list_of_stick_dicts)

    def plot_animation_all_figures(self):
        x_dims = [-20, 20]
        y_dims = [-30, 30]
        z_dims = [-25, 25]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim3d([x_dims[0],x_dims[1]])
        ax.set_xlabel('X')

        ax.set_ylim3d([y_dims[0],y_dims[1]])
        ax.set_ylabel('Y')

        ax.set_zlim3d([z_dims[0],z_dims[1]])
        ax.set_zlabel('Z')

        ax.set_title('3D Animated Skeleton')

        # Provide starting angle for the view.
        ax.view_init(10, 10)

        colors = ['b','r','g','y','m','c']
        colors = ['b', 'r', 'g', 'y', 'm', 'c','b', 'r', 'g', 'y', 'm', 'c','b', 'r', 'g', 'y', 'm', 'c','b', 'r', 'g', 'y', 'm', 'c','b', 'r', 'g', 'y', 'm', 'c','b', 'r', 'g', 'y', 'm', 'c']
        lws = [6,5.5,5,4.5,4,3.5]
        lws = [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]
        markersizes = [20,19,18,17,16,15]
        markersizes = [20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
        line = '-'
        num_figures = self.num_figures
        list_of_stick_dicts = self.list_of_stick_dicts

        bodies = []
        larms = []
        lthumbs = []
        llegs = []
        rarms = []
        rthumbs = []
        rlegs = []


        for i,stick_dicts in enumerate(list_of_stick_dicts):
            print(i)
            lw = lws[i]
            color = colors[i]
            markersize = markersizes[i]

            bodies.append(ax.plot3D(np.array([]),np.array([]),np.array([]), line, lw=lw, color=color))
            larms.append(ax.plot3D(np.array([]),np.array([]),np.array([]), line, lw=lw, color=color))
            lthumbs.append(ax.plot3D(np.array([]),np.array([]),np.array([]), line, lw=lw, color=color))
            llegs.append(ax.plot3D(np.array([]), np.array([]), np.array([]), line, lw=lw, color=color))
            rarms.append(ax.plot3D(np.array([]),np.array([]),np.array([]), line, lw=lw, color=color))
            rthumbs.append(ax.plot3D(np.array([]),np.array([]),np.array([]), line, lw=lw, color=color))
            rlegs.append(ax.plot3D(np.array([]), np.array([]), np.array([]), line, lw=lw, color=color))

        def animate(t):
            for i,stick_dicts in enumerate(list_of_stick_dicts):
                bodyx = np.array([stick_dicts["Head_End"]["CC"][t, 0], stick_dicts["Head"]["CC"][t, 0],
                                  stick_dicts["Neck1"]["CC"][t, 0], stick_dicts["Neck"]["CC"][t, 0],
                                  stick_dicts["Spine1"]["CC"][t, 0], stick_dicts["Spine"]["CC"][t, 0],
                                  stick_dicts["LowerBack"]["CC"][t, 0], stick_dicts["Hips"]["CC"][t, 0]])
                bodyy = np.array([stick_dicts["Head_End"]["CC"][t, 1], stick_dicts["Head"]["CC"][t, 1],
                                  stick_dicts["Neck1"]["CC"][t, 1], stick_dicts["Neck"]["CC"][t, 1],
                                  stick_dicts["Spine1"]["CC"][t, 1], stick_dicts["Spine"]["CC"][t, 1],
                                  stick_dicts["LowerBack"]["CC"][t, 1], stick_dicts["Hips"]["CC"][t, 1]])
                bodyz = np.array([stick_dicts["Head_End"]["CC"][t, 2], stick_dicts["Head"]["CC"][t, 2],
                                  stick_dicts["Neck1"]["CC"][t, 2], stick_dicts["Neck"]["CC"][t, 2],
                                  stick_dicts["Spine1"]["CC"][t, 2], stick_dicts["Spine"]["CC"][t, 2],
                                  stick_dicts["LowerBack"]["CC"][t, 2], stick_dicts["Hips"]["CC"][t, 2]])

                bodies
                exec('body' + str(i) + '= bodies[' + str(i) + '][0]')
                exec('body' + str(i) + '.set_data(bodyx, bodyy)')
                exec('body' + str(i) + '.set_3d_properties(bodyz)')

                llegx = np.array(
                    [stick_dicts["LeftToeBase_End"]["CC"][t, 0], stick_dicts["LeftToeBase"]["CC"][t, 0],
                     stick_dicts["LeftFoot"]["CC"][t, 0], stick_dicts["LeftLeg"]["CC"][t, 0],
                     stick_dicts["LeftUpLeg"]["CC"][t, 0], stick_dicts["LHipJoint"]["CC"][t, 0],
                     stick_dicts["Hips"]["CC"][t, 0]])
                llegy = np.array(
                    [stick_dicts["LeftToeBase_End"]["CC"][t, 1], stick_dicts["LeftToeBase"]["CC"][t, 1],
                     stick_dicts["LeftFoot"]["CC"][t, 1], stick_dicts["LeftLeg"]["CC"][t, 1],
                     stick_dicts["LeftUpLeg"]["CC"][t, 1], stick_dicts["LHipJoint"]["CC"][t, 1],
                     stick_dicts["Hips"]["CC"][t, 1]])
                llegz = np.array(
                    [stick_dicts["LeftToeBase_End"]["CC"][t, 2], stick_dicts["LeftToeBase"]["CC"][t, 2],
                     stick_dicts["LeftFoot"]["CC"][t, 2], stick_dicts["LeftLeg"]["CC"][t, 2],
                     stick_dicts["LeftUpLeg"]["CC"][t, 2], stick_dicts["LHipJoint"]["CC"][t, 2],
                     stick_dicts["Hips"]["CC"][t, 2]])

                llegs
                exec('lleg' + str(i) + '= llegs[' + str(i) + '][0]')
                exec('lleg' + str(i) + '.set_data(llegx, llegy)')
                exec('lleg' + str(i) + '.set_3d_properties(llegz)')

                larmx = np.array(
                    [stick_dicts["LeftHandIndex1_End"]["CC"][t, 0], stick_dicts["LeftHandIndex1"]["CC"][t, 0],
                     stick_dicts["LeftFingerBase"]["CC"][t, 0], stick_dicts["LeftHand"]["CC"][t, 0],
                     stick_dicts["LeftForeArm"]["CC"][t, 0], stick_dicts["LeftArm"]["CC"][t, 0],
                     stick_dicts["LeftShoulder"]["CC"][t, 0]])
                larmy = np.array(
                    [stick_dicts["LeftHandIndex1_End"]["CC"][t, 1], stick_dicts["LeftHandIndex1"]["CC"][t, 1],
                     stick_dicts["LeftFingerBase"]["CC"][t, 1], stick_dicts["LeftHand"]["CC"][t, 1],
                     stick_dicts["LeftForeArm"]["CC"][t, 1], stick_dicts["LeftArm"]["CC"][t, 1],
                     stick_dicts["LeftShoulder"]["CC"][t, 1]])
                larmz = np.array(
                    [stick_dicts["LeftHandIndex1_End"]["CC"][t, 2], stick_dicts["LeftHandIndex1"]["CC"][t, 2],
                     stick_dicts["LeftFingerBase"]["CC"][t, 2], stick_dicts["LeftHand"]["CC"][t, 2],
                     stick_dicts["LeftForeArm"]["CC"][t, 2], stick_dicts["LeftArm"]["CC"][t, 2],
                     stick_dicts["LeftShoulder"]["CC"][t, 2]])

                larms
                exec('larm' + str(i) + '= larms[' + str(i) + '][0]')
                exec('larm' + str(i) + '.set_data(larmx, larmy)')
                exec('larm' + str(i) + '.set_3d_properties(larmz)')

                lthumbx = np.array(
                    [stick_dicts["LThumb_End"]["CC"][t, 0], stick_dicts["LThumb"]["CC"][t, 0]])
                lthumby = np.array(
                    [stick_dicts["LThumb_End"]["CC"][t, 1], stick_dicts["LThumb"]["CC"][t, 1]])
                lthumbz = np.array(
                    [stick_dicts["LThumb_End"]["CC"][t, 2], stick_dicts["LThumb"]["CC"][t, 2]])

                lthumbs
                exec('lthumb' + str(i) + '= lthumbs[' + str(i) + '][0]')
                exec('lthumb' + str(i) + '.set_data(lthumbx, lthumby)')
                exec('lthumb' + str(i) + '.set_3d_properties(lthumbz)')

                rlegx = np.array(
                    [stick_dicts["RightToeBase_End"]["CC"][t, 0], stick_dicts["RightToeBase"]["CC"][t, 0],
                     stick_dicts["RightFoot"]["CC"][t, 0], stick_dicts["RightLeg"]["CC"][t, 0],
                     stick_dicts["RightUpLeg"]["CC"][t, 0], stick_dicts["LHipJoint"]["CC"][t, 0],
                     stick_dicts["Hips"]["CC"][t, 0]])
                rlegy = np.array(
                    [stick_dicts["RightToeBase_End"]["CC"][t, 1], stick_dicts["RightToeBase"]["CC"][t, 1],
                     stick_dicts["RightFoot"]["CC"][t, 1], stick_dicts["RightLeg"]["CC"][t, 1],
                     stick_dicts["RightUpLeg"]["CC"][t, 1], stick_dicts["LHipJoint"]["CC"][t, 1],
                     stick_dicts["Hips"]["CC"][t, 1]])
                rlegz = np.array(
                    [stick_dicts["RightToeBase_End"]["CC"][t, 2], stick_dicts["RightToeBase"]["CC"][t, 2],
                     stick_dicts["RightFoot"]["CC"][t, 2], stick_dicts["RightLeg"]["CC"][t, 2],
                     stick_dicts["RightUpLeg"]["CC"][t, 2], stick_dicts["LHipJoint"]["CC"][t, 2],
                     stick_dicts["Hips"]["CC"][t, 2]])

                rlegs
                exec('rleg' + str(i) + '= rlegs[' + str(i) + '][0]')
                exec('rleg' + str(i) + '.set_data(rlegx, rlegy)')
                exec('rleg' + str(i) + '.set_3d_properties(rlegz)')

                rarmx = np.array(
                    [stick_dicts["RightHandIndex1_End"]["CC"][t, 0], stick_dicts["RightHandIndex1"]["CC"][t, 0],
                     stick_dicts["RightFingerBase"]["CC"][t, 0], stick_dicts["RightHand"]["CC"][t, 0],
                     stick_dicts["RightForeArm"]["CC"][t, 0], stick_dicts["RightArm"]["CC"][t, 0],
                     stick_dicts["RightShoulder"]["CC"][t, 0]])
                rarmy = np.array(
                    [stick_dicts["RightHandIndex1_End"]["CC"][t, 1], stick_dicts["RightHandIndex1"]["CC"][t, 1],
                     stick_dicts["RightFingerBase"]["CC"][t, 1], stick_dicts["RightHand"]["CC"][t, 1],
                     stick_dicts["RightForeArm"]["CC"][t, 1], stick_dicts["RightArm"]["CC"][t, 1],
                     stick_dicts["RightShoulder"]["CC"][t, 1]])
                rarmz = np.array(
                    [stick_dicts["RightHandIndex1_End"]["CC"][t, 2], stick_dicts["RightHandIndex1"]["CC"][t, 2],
                     stick_dicts["RightFingerBase"]["CC"][t, 2], stick_dicts["RightHand"]["CC"][t, 2],
                     stick_dicts["RightForeArm"]["CC"][t, 2], stick_dicts["RightArm"]["CC"][t, 2],
                     stick_dicts["RightShoulder"]["CC"][t, 2]])

                rarms
                exec('rarm' + str(i) + '= rarms[' + str(i) + '][0]')
                exec('rarm' + str(i) + '.set_data(rarmx, rarmy)')
                exec('rarm' + str(i) + '.set_3d_properties(rarmz)')

                rthumbx = np.array(
                    [stick_dicts["LThumb_End"]["CC"][t, 0], stick_dicts["LThumb"]["CC"][t, 0]])
                rthumby = np.array(
                    [stick_dicts["LThumb_End"]["CC"][t, 1], stick_dicts["LThumb"]["CC"][t, 1]])
                rthumbz = np.array(
                    [stick_dicts["LThumb_End"]["CC"][t, 2], stick_dicts["LThumb"]["CC"][t, 2]])

                rthumbs
                exec('rthumb' + str(i) + '= rthumbs[' + str(i) + '][0]')
                exec('rthumb' + str(i) + '.set_data(rthumbx, rthumby)')
                exec('rthumb' + str(i) + '.set_3d_properties(rthumbz)')



            return exec(','.join(['body'+str(i)+', larm'+str(i)+', lthumb'+str(i)+', lleg'+str(i)+', rarm'+str(i)+', rthumb'+str(i)+', rleg'+str(i) for i in range(num_figures)]))

        return fig, animate
