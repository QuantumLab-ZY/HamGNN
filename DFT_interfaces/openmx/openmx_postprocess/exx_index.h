/*----------------------------------------------------------------------
  exx_index.h

----------------------------------------------------------------------*/
#ifndef EXX_INDEX_H_INCLUDED
#define EXX_INDEX_H_INCLUDED

int EXX_Index_Cmp_Sites(
  int iatom1,
  int iatom2,
  int icell,
  int nshell
);


int EXX_Index_Cmp_OP(
  int iatom1,
  int iatom2,
  int iatom3,
  int iatom4,
  int icell1,
  int icell2,
  int icell3,
  int nshell_op,
  int nshell_ep
);


void EXX_Index_Cell2XYZ(
  int icell,
  int nshell,
  int c[3]
);


int EXX_Index_XYZ2Cell(
  int c[3],
  int nshell
);


void EXX_Index_Cell2Cartesian(
  int icell,
  int nshell,
  const double pvec[9],
  double c[3]
);


int EXX_Index_OP_NShell(
  int           natom,   /* (IN) number of atoms */
  const double *atom_rc, /* (IN) confinement length of atoms */
  const double *atom_v,  /* (IN) atomic positions */
  const double *pvec     /* (IN) primitive translational vectors */
);


int EXX_Index_EP_NShell(
  int           natom,   /* (IN) number of atoms */
  const double *atom_rc, /* (IN) confinement length of atoms */
  const double *atom_v,  /* (IN) atomic positions */
  const double *pvec,    /* (IN) primitive translational vectors */
  double        rc_cut   /* (IN) truncation length */
);


int EXX_Index_OP(
  int           natom,   /* (IN) number of atoms */
  int           nshell,  /* (IN) numfer of shell */
  const double *atom_rc, /* (IN) confinement length of atoms */
  const double *atom_v,  /* (IN) atomic positions */
  const double *pvec,    /* (IN) primitive translational vectors */
  int          *ap_atm1, /* (OUT) array for atom1 of OPs (optional) */
  int          *ap_atm2, /* (OUT) array for atom2 of OPs (optional) */
  int          *ap_cell  /* (OUT) array for cell of OPs (optional) */
);


int EXX_Index_EQ(
  int           natom,    /* (IN) number of atoms */
  int           nshell,   /* (IN) numfer of shells */
  const double *atom_rc,  /* (IN) confinement length of atoms */
  const double *atom_v,   /* (IN) atomic positions */
  const double *pvec,     /* (IN) primitive translational vectors */
  double        rc_cut,   /* (IN) truncation length */
  int          *q_atom1,  /* (OUT) array for atom1 of EQs (optional) */
  int          *q_atom2,  /* (OUT) array for atom2 of EQs (optional) */
  int          *q_atom3,  /* (OUT) array for atom3 of EQs (optional) */
  int          *q_atom4,  /* (OUT) array for atom4 of EQs (optional) */
  int          *q_cell1,  /* (OUT) array for cell1 of EQs (optional) */
  int          *q_cell2,  /* (OUT) array for cell2 of EQs (optional) */
  int          *q_cell3   /* (OUT) array for cell3 of EQs (optional) */
);


int EXX_Index_EP(
  int           natom,    /* (IN) number of atoms */
  int           nshell,   /* (IN) numfer of shells */
  const double *atom_rc,  /* (IN) confinement length of atoms */
  const double *atom_v,   /* (IN) atomic positions */
  const double *pvec,     /* (IN) primitive translational vectors */
  double        rc_cut,   /* (IN) truncation length */
  int          *ep_atom1, /* (OUT) array for atom1 of EPs (optional) */
  int          *ep_atom2, /* (OUT) array for atom2 of EPs (optional) */
  int          *ep_cell   /* (OUT) array for cell of EPs (optioinal) */
);


int EXX_Index_NQ_Full(
  int           natom,
  int           nshell,
  const double *atom_rc,
  const double *atom_v,
  const double *pvec,
  double        rc_cut
);

int EXX_Index_NQ_Reduced(
  int           natom,
  int           nshell,
  const double *atom_rc,
  const double *atom_v,
  const double *pvec,
  double        rc_cut
);

#endif /* EXX_INDEX_H_INCLUDED */
