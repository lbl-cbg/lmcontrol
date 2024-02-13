# lmcontrol
Tools for **l**ight **m**icroscopy based bioreactor **control**

# Installation

The following commands will install the code in this repository in such a way
that will allow one to use the tools provided by said code. With that said, 
the provided sequence of commands may not suit your specific needs.
As this repository follows PEP 517 style packaging, there are many 
ways to install the software, so please use discretion and adapt as necessary.

```bash
git clone git@github.com:lbl-cbg/lmcontrol.git
cd lmcontrol
pip install -r requirements.txt
pip install .
```

# Commands

- `lmcontrol crop` - Crop light microscopy images to bounding box around single 
                     cell.
- `lmcontrol prep-viz` - Prepare package for running interactive visualizer. 
                         See `emb-viz` command for more details.
- `lmcontrol emb-viz` - Launch interactive visualizer for exploring segmented and 
                        cropped images.

## notebooks

- `Segment.ipynb` - This notebook describes how image segmentation was done.
