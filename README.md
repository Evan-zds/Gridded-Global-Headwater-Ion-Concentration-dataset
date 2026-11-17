GriGHIC v1: A Continuous Gridded Global Headwater Ion Concentration Database from 2001 to 2022

GriGHIC v1 (Gridded Global Headwater Ion Concentration database, version 1) is a 22-year (2001–2022), monthly, 0.05° × 0.05° gridded hydrochemical dataset of eight major inorganic ions (K⁺, Na⁺, Ca²⁺, Mg²⁺, Cl⁻, SO₄²⁻, NO₃⁻, and HCO₃⁻) paired with pixel-wise ensemble uncertainty estimates across global headwater catchments (upstream drainage area ≤ 10 km²).

This repository hosts:
MyData.xlsx: The harmonized, quality-controlled global in-situ headwater hydrochemical observation database compiled from peer-reviewed literature and international water-quality monitoring networks.
Model.py: The PyTorch implementation of our Physics-Informed Multi-Task Residual Neural Network, which simultaneously predicts all eight major ions while enforcing aqueous charge balance electroneutrality, carbonate equilibrium speciation, and electrical conductivity conservation within the composite loss function.

The full 22-year monthly gridded NetCDFproducts and pixel-wise uncertainty layers are permanently archived on Zenodo：
· Cation Repository: Potassium (K⁺), Sodium (Na⁺), Calcium (Ca²⁺), and Magnesium (Mg²⁺) (https://doi.org/10.5281/zenodo.22939901).
· Anion Repository: Chloride (Cl⁻), Sulfate (SO₄²⁻), Nitrate (NO₃⁻), and Bicarbonate (HCO₃⁻) (https://doi.org/10.5281/zenodo.23660469).
