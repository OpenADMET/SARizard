"""Per-flavor pretraining-target calculators.

Each flavor's self-supervised target is computed once over the shared corpus and
cached. Calculators fall into two groups by the environment they run in:

- direct-compute flavors (rdkit2d, erg, ecfp, atompair, pubchem, usrcat, whim, e3fp,
  osmordred) derive the target from the molecular graph or geometry;
- learned-model flavors (minimol, surrogate_adme) run a source model over the
  shared corpus.

Every calculator writes a plain ``target.npy`` memmap (numpy only), so the finicky or
old-Python isolated environments never need zarr. ``pack_target`` then converts that
``.npy`` into the chunked ``target.zarr`` the pretraining DataLoader consumes, in the
main ``sarizard`` environment. ``compute_target`` is the single compute entry point and
``pack_target`` the single pack entry point.
"""
