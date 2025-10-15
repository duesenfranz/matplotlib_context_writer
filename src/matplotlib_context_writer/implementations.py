
import os
from pathlib import Path
import tempfile
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from matplotlib_context_writer.interface import (
    EnteredVisualizer,
    EnterVisualizer,
    Visualizer,
)

class _SaveInDirEnteredVisualizer(EnteredVisualizer):
    """Save each `step` call into sequentially numbered PNG files.

    Example:
        >>> fig, _ = plt.subplots()
        >>> entered = _SaveInDirEnteredVisualizer(fig, Path("frames"))
        >>> entered.step()
        >>> # frame written to frames/image0000.png
    """

    def __init__(self, fig: Figure, save_dir: Path, prefix: str = "image", dpi: int=400) -> None:
        super().__init__()
        self.fig = fig
        self.save_dir = save_dir
        self.index = 0
        self.prefix = prefix
        self.dpi = dpi
    
    def step(self):
        self.fig.savefig(str(self.save_dir / f"{self.prefix}{self.index:04d}.png"), dpi=self.dpi)
        self.index += 1

class _SaveToVidEnterVisualizer(EnterVisualizer):
    """Context manager that buffers frames to disk and encodes them into a video.

    Requires `ffmpeg` to be available on the system PATH.

    Example:
        >>> fig, _ = plt.subplots()
        >>> enter = _SaveToVidEnterVisualizer(fig, Path("out.mp4"))
        >>> with enter as entered:
        ...     entered.step()
        ...     # video is written when the context exits
    """

    def __init__(self, fig: Figure, video_path: Path, fps: int=30, dpi: int=400) -> None:
        self.figure = fig
        self.video_path = video_path
        self.temp_path = None
        self.temp_path_context_manager = None
        self.prefix = "image_"
        self.fps = fps
        self.temp_video_name = "video"
        self.dpi = dpi
    
    def __enter__(self) -> _SaveInDirEnteredVisualizer:
        self.temp_path_context_manager = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_path_context_manager.__enter__()
        return _SaveInDirEnteredVisualizer(
            self.figure,
            Path(self.temp_path),
            prefix=self.prefix,
            dpi=self.dpi
        )
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.temp_path_context_manager is None:
            raise RuntimeError("Temporary directory context manager is None")
        os.system(
                f"ffmpeg -framerate {self.fps} -i {self.temp_path}/{self.prefix}%04d.png -c:v libx264 -pix_fmt yuv420p {self.temp_path}/{self.temp_video_name}.mp4"
            )
            # move the file to the current directory
        os.system(
                f"mv {self.temp_path}/{self.temp_video_name}.mp4 {self.video_path}"
            )
        self.temp_path_context_manager.__exit__(exc_type, exc_value, traceback)
    

class VideoVisualizer(Visualizer):
    """Record a Matplotlib animation to a video file using an `Enter/Entered` workflow.

    Requires `ffmpeg` to be available on the system PATH.

    Example:
        >>> from pathlib import Path
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> visualizer = VideoVisualizer(Path("demo.mp4"))
        >>> with visualizer.enter(fig) as entered:
        ...     ax.plot([0, 1], [0, 1])
        ...     entered.step()
    """

    def __init__(self, video_path: Path, fps: int = 30, dpi: int=400) -> None:
        self.video_path = video_path
        self.fps = fps
        self.dpi = dpi
    
    def enter(self, fig: Figure) -> EnterVisualizer:
        return _SaveToVidEnterVisualizer(
            fig,
            self.video_path,
            fps=self.fps,
            dpi=self.dpi,
        )

class _ShowEnteredVisualizer(EnteredVisualizer):
    """Display the figure interactively and wait for user confirmation per frame.

    Example:
        >>> fig, _ = plt.subplots()
        >>> entered = _ShowEnteredVisualizer(fig)
        >>> entered.step()  # Shows the current Matplotlib figure
    """

    def __init__(self, fig: Figure) -> None:
        super().__init__()
        self.fig = fig
    
    def step(self):
        plt.show(block=False)
        plt.pause(0.001)
        print("Press any key to continue...")
        input()

class _ShowEnterVisualizer(EnterVisualizer):
    """Wrapper that yields `_ShowEnteredVisualizer` when used as a context manager."""

    def __init__(self, fig: Figure) -> None:
        self.figure = fig
    
    def __enter__(self) -> _ShowEnteredVisualizer:
        return _ShowEnteredVisualizer(self.figure)
    
    def __exit__(self, exc_type, exc_value, traceback):
        ...

class ShowVisualizer(Visualizer):
    """Show each frame on screen and wait for user input before proceeding.

    Example:
        >>> fig, ax = plt.subplots()
        >>> visualizer = ShowVisualizer()
        >>> with visualizer.enter(fig) as entered:
        ...     ax.plot([0, 1], [0, 1])
        ...     entered.step()
    """

    def __init__(self) -> None:
        pass
    
    def enter(self, fig: Figure) -> EnterVisualizer:
        return _ShowEnterVisualizer(fig)

class _LiveVideoEnteredVisualizer(EnteredVisualizer):
    """Drive Matplotlib's event loop to create a live animation preview.

    Example:
        >>> fig, _ = plt.subplots()
        >>> entered = _LiveVideoEnteredVisualizer(fig, fps=10)
        >>> entered.step()  # Advances the live view
    """

    def __init__(self, fig: Figure, fps: float = 30) -> None:
        super().__init__()
        self.fig = fig
        self.fps = fps
    
    def step(self):
        plt.pause(1/self.fps)

class _LiveVideoEnterVisualizer(EnterVisualizer):
    """Context manager wrapper for `_LiveVideoEnteredVisualizer`."""

    def __init__(self, fig: Figure, fps: float = 30) -> None:
        self.figure = fig
        self.fps = fps
    
    def __enter__(self) -> _LiveVideoEnteredVisualizer:
        return _LiveVideoEnteredVisualizer(self.figure, self.fps)
    
    def __exit__(self, exc_type, exc_value, traceback):
        ...

class LiveVideoVisualizer(Visualizer):
    """Preview frames live by repeatedly calling `plt.pause`.

    Example:
        >>> fig, ax = plt.subplots()
        >>> visualizer = LiveVideoVisualizer(fps=15)
        >>> with visualizer.enter(fig) as entered:
        ...     ax.plot([0, 1], [0, 1])
        ...     entered.step()
    """

    def __init__(self, fps: float = 30) -> None:
        self.fps = fps
    
    def enter(self, fig: Figure) -> EnterVisualizer:
        return _LiveVideoEnterVisualizer(fig, self.fps)


def demo_video_visualizer():
    """Demonstrate recording a simple plot to a video file."""
    fig, ax = plt.subplots()
    video_path = Path.cwd() / "demo_video.mp4"
    if video_path.exists():
        video_path.unlink()

    visualizer = VideoVisualizer(video_path, fps=2, dpi=50)
    with visualizer.enter(fig) as entered:
        for x_shift in range(3):
            ax.clear()
            ax.plot([0, 1, 2], [y + x_shift * y for y in (0, 1, 0)])
            ax.set_title(f"Frame {x_shift}")
            entered.step()

    plt.close(fig)
    if video_path.exists():
        print(f"Video created at: {video_path}")
    else:
        print("Video creation failed. Ensure ffmpeg is installed and on PATH.")


def demo_show_visualizer():
    """Demonstrate how the ShowVisualizer triggers interactive display calls."""
    fig, ax = plt.subplots()
    visualizer = ShowVisualizer()
    with visualizer.enter(fig) as entered:
        for frame in range(3):
            ax.clear()
            ax.plot([0, 1, 2], [frame, frame + 0.5, 0])
            ax.set_title(f"Show frame {frame}")
            entered.step()
    plt.close(fig)
    print("ShowVisualizer demo completed.")


def demo_live_video_visualizer():
    """Demonstrate the pause timing used by LiveVideoVisualizer."""
    fig, ax = plt.subplots()
    visualizer = LiveVideoVisualizer(fps=2)
    with visualizer.enter(fig) as entered:
        for frame in range(3):
            ax.clear()
            ax.set_ylim(0, 1.5)
            ax.plot([0, 1], [0, 1 + 0.2 * frame])
            ax.set_title(f"Live frame {frame}")
            entered.step()
    plt.close(fig)
    print("LiveVideoVisualizer demo completed.")


def run_demos():
    demo_video_visualizer()
    demo_show_visualizer()
    demo_live_video_visualizer()
    print("All demos completed.")


if __name__ == "__main__":
    run_demos()
