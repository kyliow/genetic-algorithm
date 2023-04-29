import numpy
from matplotlib import patches, pyplot, figure, axes, text
from matplotlib.animation import FuncAnimation

import constants as c

from typing import Tuple, List


class Animation:
    """ 
    Class to contain initialisation and animation methods
    """
    def __init__(self):
        self.fig, self.ax = self.initialise_figure()
        self.ys, self.blocks = self.initialise_blocks()
        self.car = self.initialise_car()
        self.label = self.initialise_label()

    @staticmethod
    def initialise_figure() -> Tuple[figure.Figure, axes.Axes]:
        """ 
        Initialise the figure object
        """
        fig, ax = pyplot.subplots(1, 1, figsize=(10, 5))
        ax.set_xlim(-c.plot_axis_offset, c.N_rectangle - 1 + c.plot_axis_offset)
        ylim = (c.N_rectangle - 1) / 2 + c.plot_axis_offset
        ylim = 1
        ax.set_ylim(-ylim, ylim)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        return fig, ax

    @staticmethod
    def initialise_blocks() -> Tuple[List[float], List[patches.Rectangle]]:
        """ 
        Initialise blocks on car
        """
        ys, blocks = ([] for _ in range(2))
        for n in range(c.N_rectangle):
            # Shift from y=0
            if n == 0:
                shift = 0.25 * numpy.pi
            else:
                shift = 2 * numpy.pi * numpy.random.rand()
            # Sine movement
            ys.append(
                numpy.sin(numpy.linspace(0 + shift, 2 * numpy.pi + shift, c.N_frame))
                * (1 - c.block_height / 2)
                - c.block_height / 2
            )
            blocks.append(
                patches.Rectangle((0, 0), c.block_width, c.block_height, fc="y")
            )

        return ys, blocks

    @staticmethod
    def initialise_car() -> patches.Rectangle:
        """ 
        Initialise car object
        """
        car = patches.Rectangle((0, 0), c.block_height, c.block_width)
        return car

    def initialise_label(self) -> text.Text:
        """ 
        Initialise label in figure
        """
        label = self.ax.text(-0.05, 0, "", transform=self.ax.transAxes)
        return label

    def start_frame(self):
        for block in self.blocks:
            self.ax.add_patch(block)
        self.ax.add_patch(self.car)
        self.label.set_text("")
        all_patches = self.blocks + [self.car] + [self.label]

        return all_patches

    def animation_function(self, i, generation, position):
        for n, block in enumerate(self.blocks):
            block.set_xy([n - c.block_width / 2, self.ys[n][i]])

        self.car.set_xy([position[i] - c.block_height / 2, 0 - c.block_width / 2])

        text = f"""
            - Generation #{generation} -
            Time    : {i}
            Position: {position[i]:.3f}
        """
        self.label.set_text(text)
        all_patches = self.blocks + [self.car] + [self.label]
        return all_patches

    def animate(self, generation, times, positions, max_distance_index, save=False):
        anim = FuncAnimation(
            self.fig,
            self.animation_function,
            init_func=self.start_frame,
            frames=times[max_distance_index],
            interval=20,
            blit=True,
            fargs=(
                generation,
                positions[max_distance_index],
            ),
        )
        if save == True:
            anim.save(f"./animation/car-{generation}.gif")
        pyplot.show()
