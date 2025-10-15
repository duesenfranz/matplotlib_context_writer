
import os
from pathlib import Path
import tempfile
from typing import Optional
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

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


class _GridSnapshotEnteredVisualizer(EnteredVisualizer):
    """Capture figure snapshots for later placement in a target figure."""

    def __init__(self, source_fig: Figure, temp_dir: tempfile.TemporaryDirectory) -> None:
        super().__init__()
        self.source_fig = source_fig
        self.temp_dir = temp_dir
        self.frame_paths: list[Path] = []

    def step(self):
        frame_index = len(self.frame_paths)
        output_path = Path(self.temp_dir.name) / f"frame_{frame_index:04d}.png"
        self.source_fig.savefig(output_path)
        self.frame_paths.append(output_path)


class _GridSnapshotEnterVisualizer(EnterVisualizer):
    """Context manager that prepares a grid of snapshots when exiting."""

    def __init__(self, source_fig: Figure, target_fig: Figure, count: int, orientation: str) -> None:
        if count < 2:
            raise ValueError("count must be at least 2 to include first and last frames.")
        orientation_upper = orientation.upper()
        if orientation_upper not in {"ROW", "COLUMN"}:
            raise ValueError("orientation must be either 'ROW' or 'COLUMN'.")
        self.source_fig = source_fig
        self.target_fig = target_fig
        self.count = count
        self.orientation = orientation_upper
        self.temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.entered: Optional[_GridSnapshotEnteredVisualizer] = None

    def __enter__(self) -> _GridSnapshotEnteredVisualizer:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.entered = _GridSnapshotEnteredVisualizer(self.source_fig, self.temp_dir)
        return self.entered

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self.entered is None:
                return
            frame_paths = self.entered.frame_paths
            if not frame_paths:
                return
            selected_paths = self._select_frames(frame_paths)
            self._render_to_target(selected_paths)
        finally:
            if self.temp_dir is not None:
                self.temp_dir.cleanup()

    def _select_frames(self, frame_paths: list[Path]) -> list[Path]:
        total = len(frame_paths)
        desired = min(self.count, total)
        if desired == 1:
            return [frame_paths[-1]]
        if desired == total:
            return frame_paths
        indices: list[int] = []
        for idx in range(desired):
            fraction = idx / (desired - 1)
            computed = int(round(fraction * (total - 1)))
            if not indices or computed != indices[-1]:
                indices.append(computed)
        while len(indices) < desired:
            for candidate in range(total):
                if candidate not in indices:
                    indices.append(candidate)
                if len(indices) == desired:
                    break
        indices = sorted(indices[:desired])
        return [frame_paths[i] for i in indices]

    def _render_to_target(self, selected_paths: list[Path]) -> None:
        self.target_fig.clear()
        n_frames = len(selected_paths)
        if self.orientation == "ROW":
            axes = self.target_fig.subplots(1, n_frames)
        else:
            axes = self.target_fig.subplots(n_frames, 1)
        if hasattr(axes, "flat"):
            axes_list = list(axes.flat)
        elif isinstance(axes, (list, tuple)):
            axes_list = list(axes)
        else:
            axes_list = [axes]
        for ax, image_path in zip(axes_list, selected_paths):
            ax.imshow(mpimg.imread(image_path))
            ax.axis("off")
        self.target_fig.tight_layout()
        self.target_fig.canvas.draw_idle()


class GridSnapshotVisualizer(Visualizer):
    """Summarize animation progress by arranging selected frames in a grid."""

    def __init__(self, target_fig: Figure, count: int, orientation: str = "ROW") -> None:
        self.target_fig = target_fig
        self.count = count
        self.orientation = orientation

    def enter(self, fig: Figure) -> EnterVisualizer:
        return _GridSnapshotEnterVisualizer(fig, self.target_fig, self.count, self.orientation)


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


def demo_grid_snapshot_visualizer():
    """Demonstrate summarizing key frames in a dedicated figure."""
    source_fig, source_ax = plt.subplots()
    target_fig = plt.figure(figsize=(8, 2))
    visualizer = GridSnapshotVisualizer(target_fig, count=4, orientation="ROW")
    with visualizer.enter(source_fig) as entered:
        for frame in range(6):
            source_ax.clear()
            source_ax.plot([0, 1, 2], [frame, frame + 0.5, frame])
            source_ax.set_ylim(0, 6)
            source_ax.set_title(f"Source frame {frame}")
            entered.step()
    plt.close(source_fig)
    target_fig.suptitle("GridSnapshotVisualizer Demo")
    target_fig.canvas.draw_idle()
    plt.show(block=False)
    plt.pause(2)
    plt.close(target_fig)
    print("GridSnapshotVisualizer demo completed.")


def run_demos():
    demo_video_visualizer()
    demo_show_visualizer()
    demo_live_video_visualizer()
    demo_grid_snapshot_visualizer()
    print("All demos completed.")


if __name__ == "__main__":
    run_demos()
