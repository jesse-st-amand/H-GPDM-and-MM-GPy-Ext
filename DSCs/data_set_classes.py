from .data_set_class_base import DataSetClassMovementBase
from .graphics.stick_graphic_3D import StickGraphic
from .graphics.interactive_stick_figures_3D import InteractiveFigures
from .graphics.stick_graphic_3D import StickGraphicCMU
from .graphics.interactive_stick_figures_3D import InteractiveFiguresCMU
import os
class Bimanual3D(DataSetClassMovementBase):
    def __init__(self, seq_len, full_num_seqs_per_subj_per_act, num_folds=1, fold_num='all'):
        super().__init__('Bimanual 3D', seq_len)
        self.initialize_dataset(
            fold_num=fold_num,
            num_folds=num_folds,
            full_num_seqs_per_subj_per_act=full_num_seqs_per_subj_per_act,
            graphic=StickGraphic(),
            IFs_func=InteractiveFigures,
            actions=[
                'box_lift',
                'box_turn_cw_forward',
                'box_lateral_left',
                'bread_cutting_right',
                'jar_opening_right_open'
            ],
            subjects=['Lucas', 'Jens', 'Jana', 'Alex', 'Lisa'],
            score_EPs_ff=['LeftHand', 'RightHand'],
            score_EPs_fb=['LeftHand', 'RightHand'],
        )
        self.X_init = None  # Ensure X_init is initialized

    def sequence_actions(self):
        npy_dir = os.path.join(os.path.dirname(__file__), "graphics", "npy", "3D")
        return self._sequence_actions(npy_dir, 'PCs_', self.num_tps)





class MovementsCMU(DataSetClassMovementBase):
    def __init__(self, seq_len, full_num_seqs_per_subj_per_act, num_folds=1, fold_num='all'):
        super().__init__('Movements CMU', seq_len)
        self.initialize_dataset(
            fold_num=fold_num,
            num_folds=num_folds,
            full_num_seqs_per_subj_per_act=full_num_seqs_per_subj_per_act,
            graphic=StickGraphicCMU(),
            IFs_func=InteractiveFiguresCMU,
            actions=[
                'bend_down',
                'soccer_kick',
                'jump',
                'breaststroke',
                'flystroke',
                'jump_side',
                'left_front_kick',
                'left_lunges',
                'left_punches'
            ],
            subjects=['01'],
            score_EPs_ff=['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot', 'Hips'],
            score_EPs_fb=['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot', 'Hips'],
        )
        self.X_init = None  # Ensure X_init is initialized

    def sequence_actions(self):
        npy_dir = os.path.join(os.path.dirname(__file__), "graphics", "npy", "CMU")
        return self._sequence_actions(npy_dir, 'PCs_', self.num_tps)
