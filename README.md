# Environmental Risk Metrics

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)

Calculate environmental risk metrics for a given polygon using advanced geospatial and data processing tools.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the API](#running-the-api)
  - [Using Jupyter Notebooks](#using-jupyter-notebooks)
- [Data Resources](#data-resources)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **NDVI Calculation:** Compute Normalized Difference Vegetation Index (NDVI) values for specified polygons.
- **Sentinel-2 Integration:** Load and process Sentinel-2 satellite data for various spectral bands.
- **Interactive Notebooks:** Utilize Jupyter notebooks for data analysis and visualization.
- **Comprehensive Soil Data:** Incorporates detailed soil type information for accurate risk assessment.
- **Protected Areas:** Get nearest Ramsar protected sites for a given geometry
- **Social Indices:** Get Global Witness data for the countries containing or intersecting the given geometry
- **Endangered Species:** Get endangered species data for the countries containing or intersecting the given geometry
- **Climate Data:** Get climate data for the countries containing or intersecting the given geometry

## Getting Started

### Prerequisites

- Python 3.12+
- [Git](https://git-scm.com/)

### Installation

1. **Clone the Repository**

   ```bash
   pip install environmental-risk-metrics
   ```

## Examples

### Using Jupyter Notebooks

Interactive analysis can be performed using the provided Jupyter notebooks.

1. **Navigate to the Notebooks Directory**

   ```bash
   cd notebooks
   ```

2. **Launch Jupyter Notebook**

   ```bash
   jupyter notebook
   ```

3. **Open and Run the Desired Notebook**

   For example, open `01 - all_metrics.ipynb` to explore environmental risk metrics calculations.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add some feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

Developed by [Thimm](mailto:thimm@regenrate.com). For any inquiries or feedback, please reach out via email.