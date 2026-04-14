# Objective
Refactor the T1 calibration animation so the exported MP4 shows only the left 3D calibration view and the right synthetic image view, with both panels enlarged for clearer presentation. Keep the separate intrinsics PNG output unchanged.

# User Value
- The T1 animation becomes easier to read because the MP4 no longer wastes space on a third intrinsics panel that already has its own PNG.
- The two remaining panels can be rendered larger, which improves visual clarity and export quality.
- The change stays localized to T1 video rendering and does not alter the calibration computation or the saved figure outputs.

# Constraints
- The public MATLAB deliverable must remain stable.
- The T1 calibration computation, saved PNG outputs, and numeric metrics must not change.
- The MP4 should contain only the left and right panels and should remain suitable for the existing smoke test.
- Other tasks (`T2`, `T3`) must remain untouched.

# Assumptions
- [ASSUMPTION] The existing intrinsics PNG already covers the numerical comparison need, so the MP4 does not need to repeat it.
- [ASSUMPTION] A separate 1x2 video-only figure is the safest way to enlarge the panels without affecting the static 2x2 summary figure.
- [ASSUMPTION] MATLAB `getframe` will continue to capture the larger dedicated video figure correctly in the current environment.

# Affected Files
- `matlab/src/t1_virtual_calibration.m`
- `matlab/tests/smoke_test.m` [only if validation reveals a path or size regression]

# Steps
1. Add a video-only 1x2 figure for T1 animation with the board path on the left and the synthetic image on the right.
2. Keep the existing 2x2 static figure for saved PNG outputs, including the intrinsics panel.
3. Increase the video figure size so both panels render at higher resolution.
4. Validate that the T1 MP4 still exports and that the overall simulation smoke test passes.

# Validation
- Run `run_demo('showFigures', false, 'saveVideos', true, 'saveSummary', false)` in MATLAB batch.
- Confirm the T1 MP4 is generated and the static PNG outputs are still present.
- Check that T1/T2/T3 summaries still complete without errors.

# Risks
- If the new video-only figure is too large, `getframe` may capture a frame size that makes the MP4 heavier than expected.
- If the video figure and static figure share handles incorrectly, the saved PNGs could accidentally change.
- If the animation loop still references the old 2x2 axes, the MP4 could continue to include the intrinsics panel or fail to render.
- Large figure sizes can expose platform-specific rendering differences in MATLAB batch mode.

# Rollback Notes
- If the 1x2 video layout causes any regression, revert only the T1 video branch and keep the static figure path untouched.
- If the larger export size becomes problematic, shrink the video figure dimensions before changing the control logic.
- If any smoke test fails, restore the previous 2x2 video capture while keeping the saved PNG outputs intact.

# Overlooked Risks
- The T1 animation may still look cramped if the left 3D view uses too much text or annotation space.
- The right image panel can become visually noisy if the point overlay is too large at higher resolution.
- The legend-free layout can still look unbalanced if the axes limits or camera frame labels are not tuned for the larger export.
