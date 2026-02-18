# TTCBF: A Truncated Taylor Control Barrier Function for High-Order Safety Constraints
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/bassamlab/ttcbf/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2408.07644-b31b1b.svg)](https://arxiv.org/abs/2601.15196)

**Abstract**: Control Barrier Functions (CBFs) enforce safety by rendering a prescribed safe set forward invariant. However, standard CBFs are limited to safety constraints with relative degree one, while High-Order CBF (HOCBF) methods address higher relative degree at the cost of introducing a chain of auxiliary functions and multiple class K functions whose tuning scales with the relative degree. In this paper, we introduce a Truncated Taylor Control Barrier Function (TTCBF), which generalizes standard discrete-time CBFs to consider high-order safety constraints and requires only one class K function, independent of the relative degree. We also propose an adaptive variant, adaptive TTCBF (aTTCBF), that optimizes an online gain on the class K function to improve adaptability, while requiring fewer control design parameters than existing adaptive HOCBF variants. Numerical experiments in a relative-degree-six spring-mass system and a cluttered corridor navigation validate the above theoretical findings.

## Install
- Requirements
  - Python 3.11 (other versions may also work)
  - All required Python packages are listed in `requirements.txt`

- Create a Virtual Environment (Recommended)
  ```bash
  # Create environment
  conda create -n ttcbf python=3.11 -y

  # Activate environment
  conda activate ttcbf

  # Install dependencies
  pip install -r requirements.txt
  ```

## How to Use
- For the spring-mass system, run `run_spring_mass.py`, or run `plot_spring_mass.py` directly using the saved data.
- For corridor navigation, run `run_corridor_1.py` to compare aTTCBF with TTCBF, or run `plot_corridor_1.py` directly using the saved data. Run `run_corridor_2.py` to compare aTTCBF with PACBF and RACBF, or run `plot_corridor_2.py` directly using the saved data.
  
## Simulation Videos

- Spring-Mass System: Relative Degree Six
<table>
  <tr>
    <td align="center" width="49%">
      <b>Our aTTCBF</b><br>
      <img src="video/video_spring_mass_ttcbf.gif" alt="TTCBF (Linear) GIF" width="100%">
    </td>
    <td align="center" width="49%">
      <b>Nominal Controller</b><br>
      <img src="video/video_spring_mass_none.gif" alt="TTCBF (Exponential) GIF" width="100%">
    </td>
  </tr>
</table>


- Corridor Navigation: Comparing aTTCBF with TTCBF
<table>
  <tr>
    <td align="center" width="33%">
      <b>TTCBF and aTTCBF (Linear Class K)</b><br>
      <img src="video/video_all_ttcbf_LINEAR.gif" alt="TTCBF (Linear) GIF" width="100%">
    </td>
    <td align="center" width="33%">
      <b>TTCBF and aTTCBF (Exponential Class K)</b><br>
      <img src="video/video_all_ttcbf_EXP.gif" alt="TTCBF (Exponential) GIF" width="100%">
    </td>
    <td align="center" width="33%">
      <b>TTCBF and aTTCBF (Rational Class K)</b><br>
      <img src="video/video_all_ttcbf_RATIONAL.gif" alt="TTCBF (Rational) GIF" width="100%">
    </td>
  </tr>
  <!-- <tr>
    <td align="center" width="33%">
      <b>Adaptive CBFs</b><br>
      <img src="video/video_all_acbfs.gif" alt="Adaptive CBF Baselines (aCBFs) GIF" width="100%">
    </td>
    <td width="33%"></td>
    <td width="33%"></td>
  </tr> -->
</table>

- Corridor Navigation: Comparing Our aTTCBF with PACBF and RACBF:
<table>
  <tr>
    <td align="center">
      <b>Adaptive CBFs</b><br>
      <img src="video/video_all_acbfs.gif"
           alt="Adaptive CBF Baselines (aCBFs) GIF"
           style="width:50%;">
    </td>
  </tr>
</table>
