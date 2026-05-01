#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <libgen.h>
#include <sys/stat.h>
#include <unistd.h>
#include <ctype.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define BOHR_TO_ANGSTROM 0.529177249
#define MAX_SPECIES 64
#define MAX_ATOMS 1024
#define MAX_L 6
#define MAX_MUL 6
#define MAX_NAME_LEN 64
#define MAX_MESH 3000

// ============================================================
// Data Structures
// ============================================================

typedef struct {
    char name[MAX_NAME_LEN];
    char basis_name[MAX_NAME_LEN];
    char elem_name[MAX_NAME_LEN];
    int Lmax;
    int Mul;
    int Mesh;
    double *RV;
    double ***RWF;    // RWF[L][Mul][mesh]
} SpeciesPAO;

typedef struct {
    int species_idx;
    double x, y, z;       // Bohr
    int norbs;            // orbitals per atom
    int zeta[MAX_L + 1];
} Atom;

typedef struct {
    int species_count;
    SpeciesPAO species[MAX_SPECIES];
    int atom_count;
    Atom atoms[MAX_ATOMS];
    int Ngrid1, Ngrid2, Ngrid3;
    double tv[4][4];       // lattice vectors
    char data_path[1024];
    double energycutoff;   // scf.energycutoff in Rydberg
    int has_fixed_grid;    /* 1 if Ngrid1/2/3 are explicitly set in dat */
    /* Map element name -> basis name (for PAO file lookup) */
    char elem_basis[MAX_SPECIES][MAX_NAME_LEN];
    int elem_count;
} System;

// ============================================================
// Grid calculation from energy cutoff
// ============================================================

static int is_fft_friendly(int n) {
    int tmp = n;
    while (tmp % 2 == 0) tmp /= 2;
    while (tmp % 3 == 0) tmp /= 3;
    while (tmp % 5 == 0) tmp /= 5;
    return tmp == 1;
}

static void calc_grid_from_ecutoff(double tv[4][4], double ecut,
                                   int *Ngrid1, int *Ngrid2, int *Ngrid3) {
    double a = M_PI / sqrt(ecut);
    int grids[4];

    for (int i = 1; i <= 3; i++) {
        double len = sqrt(tv[i][1]*tv[i][1] + tv[i][2]*tv[i][2] + tv[i][3]*tv[i][3]);
        int n = (int)ceil(len / a);
        while (!is_fft_friendly(n)) {
            n++;
        }
        grids[i] = n;
    }
    *Ngrid1 = grids[1];
    *Ngrid2 = grids[2];
    *Ngrid3 = grids[3];
}

static int elem_name_to_idx(System *sys, const char *name) {
    for (int i = 0; i < sys->elem_count; i++) {
        if (strcmp(sys->elem_basis[i], name) == 0) return i;
    }
    /* Return first unused slot; caller should check bounds */
    if (sys->elem_count < MAX_SPECIES) {
        int idx = sys->elem_count;
        strncpy(sys->elem_basis[idx], name, MAX_NAME_LEN - 1);
        sys->elem_basis[idx][MAX_NAME_LEN - 1] = '\0';
        sys->elem_count++;
        return idx;
    }
    return -1;
}

// ============================================================
// Math: xyz2spherical (from OpenMX xyz2spherical.c)
// ============================================================

static void xyz2spherical(double dx, double dy, double dz,
                          double *R, double *theta, double *phi)
{
    double r, r1, dum, dum1;
    const double Min_r = 1e-14;

    dum = dx*dx + dy*dy;
    r = sqrt(dum + dz*dz);
    r1 = sqrt(dum);

    *R = r;
    if (Min_r <= r) {
        if (r < fabs(dz))
            dum1 = (dz >= 0 ? 1.0 : -1.0);
        else
            dum1 = dz / r;
        *theta = acos(dum1);

        if (Min_r <= r1) {
            if (0.0 <= dx) {
                if (r1 < fabs(dy))
                    dum1 = (dy >= 0 ? 1.0 : -1.0);
                else
                    dum1 = dy / r1;
                *phi = asin(dum1);
            } else {
                if (r1 < fabs(dy))
                    dum1 = (dy >= 0 ? 1.0 : -1.0);
                else
                    dum1 = dy / r1;
                *phi = M_PI - asin(dum1);
            }
        } else {
            *phi = 0.0;
        }
    } else {
        *theta = 0.5 * M_PI;
        *phi = 0.0;
    }
}

// ============================================================
// Math: AngularF - Real Spherical Harmonics (from AngularF.c)
// ============================================================

static double AngularF(int l, int m, double Q, double P)
{
    double siQ = sin(Q), coQ = cos(Q);
    double siP = sin(P), coP = cos(P);

    switch (l) {
        case 0:
            return 0.5 / sqrt(M_PI);

        case 1:
            switch (m) {
                case 0: return 0.5 * sqrt(3.0/M_PI) * siQ * coP;
                case 1: return 0.5 * sqrt(3.0/M_PI) * siQ * siP;
                case 2: return 0.5 * sqrt(3.0/M_PI) * coQ;
                default: return 0.0;
            }

        case 2:
            switch (m) {
                case 0: return 0.94617469575756 * coQ*coQ - 0.31539156525252;
                case 1: return 0.54627421529604 * siQ*siQ * (1.0 - 2.0*siP*siP);
                case 2: return 1.09254843059208 * siQ*siQ * siP*coP;
                case 3: return 1.09254843059208 * siQ*coQ * coP;
                case 4: return 1.09254843059208 * siQ*coQ * siP;
                default: return 0.0;
            }

        case 3:
            switch (m) {
                case 0: return 0.373176332590116 * (5.0*coQ*coQ*coQ - 3.0*coQ);
                case 1: return 0.457045799464466 * coP*siQ * (5.0*coQ*coQ - 1.0);
                case 2: return 0.457045799464466 * siP*siQ * (5.0*coQ*coQ - 1.0);
                case 3: return 1.44530572132028 * siQ*siQ*coQ * (coP*coP - siP*siP);
                case 4: return 2.89061144264055 * siQ*siQ*coQ * siP*coP;
                case 5: return 0.590043589926644 * siQ*siQ*siQ * (4.0*coP*coP*coP - 3.0*coP);
                case 6: return 0.590043589926644 * siQ*siQ*siQ * (3.0*siP - 4.0*siP*siP*siP);
                default: return 0.0;
            }

        case 4: case 5: case 6:
            fprintf(stderr, "Warning: AngularF L=%d needs ComplexSH (not implemented)\n", l);
            return 0.0;

        default: return 0.0;
    }
}

// ============================================================
// Math: PhiF - Hermite cubic spline interpolation (from PhiF.c)
// ============================================================

static double PhiF(double R, const double *phi0, const double *RV, int Grid_Num)
{
    int mp_min, mp_max, m;
    double h1, h2, h3, f1, f2, f3, f4, g1, g2, x1, x2, y1, y2, f, df, rm, a, b;

    if (R > RV[Grid_Num - 1]) return 0.0;

    if (R < RV[0]) {
        /* Parabolic extrapolation to origin (matching PhiF.c) */
        m = 4;
        if (m < 2) m = 2;
        else if (m >= Grid_Num - 1) m = Grid_Num - 2;

        rm = RV[m];

        h1 = RV[m-1] - RV[m-2];
        h2 = RV[m]   - RV[m-1];
        h3 = RV[m+1] - RV[m];

        f1 = phi0[m-2];
        f2 = phi0[m-1];
        f3 = phi0[m];
        f4 = phi0[m+1];

        g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1) / (h1+h2);
        g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2) / (h2+h3);

        x1 = rm - RV[m-1];
        x2 = rm - RV[m];
        y1 = x1 / h2;
        y2 = x2 / h2;

        f =  y2*y2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
           + y1*y1*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

        df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
           + y2*y2*(2.0*f2 + h2*g1)/h2
           + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
           - y1*y1*(2.0*f3 - h2*g2)/h2;

        /* Parabolic fit: a*R^2 + b with df/dR(R=rm) = df */
        a = 0.5 * df / rm;
        b = f - a * rm * rm;
        return a * R * R + b;
    }

    mp_min = 0;
    mp_max = Grid_Num - 1;
    do {
        m = (mp_min + mp_max) / 2;
        if (RV[m] < R) mp_min = m;
        else mp_max = m;
    } while ((mp_max - mp_min) != 1);
    m = mp_max;

    if (m < 2) m = 2;
    else if (m >= Grid_Num - 1) m = Grid_Num - 2;

    h1 = RV[m-1] - RV[m-2];
    h2 = RV[m]   - RV[m-1];
    h3 = RV[m+1] - RV[m];

    f1 = phi0[m-2];
    f2 = phi0[m-1];
    f3 = phi0[m];
    f4 = phi0[m+1];

    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1) / (h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2) / (h2+h3);

    x1 = R - RV[m-1];
    x2 = R - RV[m];
    y1 = x1 / h2;
    y2 = x2 / h2;

    return y2*y2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
         + y1*y1*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);
}

// ============================================================
// Parser: .pao file loader
// ============================================================

static void free_species_pao(SpeciesPAO *sp) {
    if (sp->RWF) {
        for (int l = 0; l <= sp->Lmax; l++) {
            if (sp->RWF[l]) {
                for (int m = 0; m < sp->Mul; m++) {
                    free(sp->RWF[l][m]);
                }
                free(sp->RWF[l]);
            }
        }
        free(sp->RWF);
    }
    free(sp->RV);
    sp->RWF = NULL;
    sp->RV = NULL;
}

static int load_species_pao(const char *pao_path, SpeciesPAO *sp) {
    FILE *fp = fopen(pao_path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open PAO file: %s\n", pao_path);
        return -1;
    }

    memset(sp, 0, sizeof(*sp));
    strncpy(sp->name, basename((char*)pao_path), MAX_NAME_LEN - 1);
    char *dot = strrchr(sp->name, '.');
    if (dot) *dot = '\0';

    char line[4096];
    /* Phase 1: scan for Lmax, Mul, Mesh */
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "PAO.Lmax", 8) == 0) {
            char *p = line + 8;
            while (*p == ' ' || *p == '\t') p++;
            sscanf(p, "%d", &sp->Lmax);
        }
        if (strncmp(line, "PAO.Mul", 7) == 0) {
            char *p = line + 7;
            while (*p == ' ' || *p == '\t') p++;
            sscanf(p, "%d", &sp->Mul);
        }
        if (strstr(line, "grid.num.output")) {
            char *p = strstr(line, "grid.num.output") + 15;
            while (*p == ' ' || *p == '\t') p++;
            sscanf(p, "%d", &sp->Mesh);
        }
    }

    if (sp->Lmax < 0 || sp->Mul < 1 || sp->Mesh < 1) {
        fprintf(stderr, "Error: Invalid PAO header in %s (Lmax=%d Mul=%d Mesh=%d)\n",
                pao_path, sp->Lmax, sp->Mul, sp->Mesh);
        fflush(stderr); fclose(fp);
        return -1;
    }

    /* Allocate */
    sp->RV = (double*)calloc(sp->Mesh, sizeof(double));
    sp->RWF = (double***)calloc(sp->Lmax + 1, sizeof(double**));
    for (int l = 0; l <= sp->Lmax; l++) {
        sp->RWF[l] = (double**)calloc(sp->Mul, sizeof(double*));
        for (int m = 0; m < sp->Mul; m++) {
            sp->RWF[l][m] = (double*)calloc(sp->Mesh, sizeof(double));
        }
    }

    /* Phase 2: read orbital data blocks */
    fseek(fp, 0, SEEK_SET);

    for (int L = 0; L <= sp->Lmax; L++) {
        char open_tag[128];
        snprintf(open_tag, sizeof(open_tag), "<pseudo.atomic.orbitals.L=%d", L);

        /* Find opening tag */
        int found = 0;
        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, open_tag, strlen(open_tag)) == 0) {
                found = 1;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "Error: Missing tag '%s' in %s\n", open_tag, pao_path);
            fflush(stderr); fclose(fp);
            return -1;
        }

        /* Read Mesh lines */
        for (int i = 0; i < sp->Mesh; i++) {
            if (!fgets(line, sizeof(line), fp)) {
                fprintf(stderr, "Error: EOF in PAO L=%d block\n", L);
                fflush(stderr); fclose(fp);
                return -1;
            }
            /* Format: XV  RV  PAO[L][0] ... PAO[L][Mul-1] */
            char *ptr = line;
            /* Skip XV */
            strtod(ptr, &ptr);
            while (*ptr == ' ' || *ptr == '\t') ptr++;
            /* Read RV */
            sp->RV[i] = strtod(ptr, &ptr);
            while (*ptr == ' ' || *ptr == '\t') ptr++;
            /* Read Mul PAO values */
            for (int m = 0; m < sp->Mul; m++) {
                sp->RWF[L][m][i] = strtod(ptr, &ptr);
                while (*ptr == ' ' || *ptr == '\t') ptr++;
            }
        }

        /* Skip to closing tag */
        char close_tag[128];
        snprintf(close_tag, sizeof(close_tag), "pseudo.atomic.orbitals.L=%d>", L);
        while (fgets(line, sizeof(line), fp)) {
            if (strstr(line, close_tag)) break;
        }
    }

    fflush(stderr); fclose(fp);
    return 0;
}

static void free_system(System *sys) {
    for (int i = 0; i < sys->species_count; i++) {
        free_species_pao(&sys->species[i]);
    }
}

/* Look up species by name, load if not cached */
static int resolve_species(System *sys, const char *species_name, const char *basis_name) {
    for (int i = 0; i < sys->species_count; i++) {
        if (strcmp(sys->species[i].elem_name, species_name) == 0) {
            return i;
        }
    }
    if (sys->species_count >= MAX_SPECIES) {
        fprintf(stderr, "Error: Too many species\n");
        return -1;
    }
    int idx = sys->species_count;

    char basis_truncated[MAX_NAME_LEN];
    strncpy(basis_truncated, basis_name ? basis_name : species_name, MAX_NAME_LEN - 1);
    basis_truncated[MAX_NAME_LEN - 1] = '\0';
    char *dash = strchr(basis_truncated, '-');
    if (dash) *dash = '\0';

    char pao_path[2048];
    snprintf(pao_path, sizeof(pao_path), "%s/PAO/%s.pao", sys->data_path, basis_truncated);
    int ret = load_species_pao(pao_path, &sys->species[idx]);
    if (ret != 0) return -1;
    strncpy(sys->species[idx].name, species_name, MAX_NAME_LEN - 1);
    strncpy(sys->species[idx].elem_name, species_name, MAX_NAME_LEN - 1);
    strncpy(sys->species[idx].basis_name, basis_truncated, MAX_NAME_LEN - 1);
    sys->species_count++;
    return idx;
}

static int count_norbs_for_species(const SpeciesPAO *sp) {
    int n = 0;
    for (int L = 0; L <= sp->Lmax; L++) {
        int m_count;
        if (L == 0) m_count = 1;
        else if (L == 1) m_count = 3;
        else if (L == 2) m_count = 5;
        else if (L == 3) m_count = 7;
        else m_count = 2 * L + 1;
        n += m_count * sp->Mul;
    }
    return n;
}

static void parse_zeta_counts(const char *basis_name, int *zeta) {
    const char *p = strchr(basis_name, '-');
    if (!p) return;
    p++;
    while (*p) {
        char orb = 0;
        if (*p >= 'a' && *p <= 'z') { orb = *p; p++; }
        if (orb && *p >= '0' && *p <= '9') {
            int val = *p++ - '0';
            while (*p >= '0' && *p <= '9') val = val * 10 + (*p++ - '0');
            int l = (orb == 's') ? 0 : (orb == 'p') ? 1 : (orb == 'd') ? 2 : (orb == 'f') ? 3 : -1;
            if (l >= 0 && l <= MAX_L) zeta[l] = val;
        } else if (*p) {
            p++;
        }
    }
}

static int count_norbs_from_zeta(const int *zeta, int Lmax) {
    int n = 0;
    for (int L = 0; L <= Lmax; L++) {
        int m_count;
        if (L == 0) m_count = 1;
        else if (L == 1) m_count = 3;
        else if (L == 2) m_count = 5;
        else if (L == 3) m_count = 7;
        else m_count = 2 * L + 1;
        n += m_count * zeta[L];
    }
    return n;
}

typedef struct {
    char elem_name[MAX_NAME_LEN];
    char basis_name[MAX_NAME_LEN];
} ElemBasisEntry;

static int parse_dat(const char *dat_path, System *sys) {
    FILE *fp = fopen(dat_path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open dat file: %s\n", dat_path);
        return -1;
    }

    memset(sys, 0, sizeof(*sys));

    char line[4096];
    char coord_unit[16] = "Ang";
    double tv_ang[4][4];
    int in_atoms_coords = 0;
    int in_unit_vectors = 0;
    int unit_vectors_read = 0;
    int atoms_coords_read = 0;
    int in_definition = 0;

    ElemBasisEntry definitions[MAX_SPECIES];
    int def_count = 0;

    while (fgets(line, sizeof(line), fp)) {
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';

        if (line[0] == '\0' || line[0] == '#') continue;

        if (strstr(line, "DATA.PATH")) {
            char *p = strstr(line, "DATA.PATH") + 9;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '"' || *p == '\'') p++;
            char *end = p;
            while (*end && *end != '"' && *end != '\'' && *end != ' ' && *end != '\t') end++;
            *end = '\0';
            strncpy(sys->data_path, p, sizeof(sys->data_path) - 1);
        }

        if (strstr(line, "scf.energycutoff")) {
            char *p = strstr(line, "scf.energycutoff") + 16;
            while (*p == ' ' || *p == '\t') p++;
            sscanf(p, "%lf", &sys->energycutoff);
            continue;
        }

        if (strstr(line, "scf.Ngrid1")) {
            char *p = strstr(line, "scf.Ngrid1") + 10;
            while (*p == ' ' || *p == '\t') p++;
            sscanf(p, "%d", &sys->Ngrid1);
            continue;
        }
        if (strstr(line, "scf.Ngrid2")) {
            char *p = strstr(line, "scf.Ngrid2") + 10;
            while (*p == ' ' || *p == '\t') p++;
            sscanf(p, "%d", &sys->Ngrid2);
            continue;
        }
        if (strstr(line, "scf.Ngrid3")) {
            char *p = strstr(line, "scf.Ngrid3") + 10;
            while (*p == ' ' || *p == '\t') p++;
            sscanf(p, "%d", &sys->Ngrid3);
            continue;
        }

        if (strstr(line, "<Definition.of.Atomic.Species")) {
            in_definition = 1;
            continue;
        }
        if (in_definition && strstr(line, "Definition.of.Atomic.Species>")) {
            in_definition = 0;
            continue;
        }
        if (in_definition) {
            char tok1[64], tok2[64], tok3[64];
            int sp_idx;
            double mass;

            char *p = line;
            while (*p == ' ' || *p == '\t') p++;

            if (isdigit((unsigned char)*p)) {
                int n = sscanf(line, "%d %63s %63s %63s %lf", &sp_idx, tok1, tok2, tok3, &mass);
                if (n >= 3 && def_count < MAX_SPECIES) {
                    strncpy(definitions[def_count].elem_name, tok1, MAX_NAME_LEN - 1);
                    strncpy(definitions[def_count].basis_name, tok2, MAX_NAME_LEN - 1);
                    def_count++;
                }
            } else {
                int n = sscanf(p, "%63s %63s %63s %lf", tok1, tok2, tok3, &mass);
                if (n >= 2 && def_count < MAX_SPECIES) {
                    strncpy(definitions[def_count].elem_name, tok1, MAX_NAME_LEN - 1);
                    strncpy(definitions[def_count].basis_name, tok2, MAX_NAME_LEN - 1);
                    def_count++;
                }
            }
            continue;
        }

        if (strstr(line, "Atoms.SpeciesAndCoordinates.Unit")) {
            char *p = strstr(line, "Atoms.SpeciesAndCoordinates.Unit") + 32;
            while (*p == ' ' || *p == '\t') p++;
            if (strncmp(p, "FRAC", 4) == 0) {
                strcpy(coord_unit, "FRAC");
            } else if (strncmp(p, "AU", 2) == 0) {
                strcpy(coord_unit, "AU");
            } else {
                strcpy(coord_unit, "Ang");
            }
        }

        if (strstr(line, "Atoms.UnitVectors.Unit")) {
            char *p = strstr(line, "Atoms.UnitVectors.Unit") + 22;
            while (*p == ' ' || *p == '\t') p++;
            (void)p;
        }


        if (strstr(line, "<Atoms.SpeciesAndCoordinates") || strstr(line, " <Atoms.SpeciesAndCoordinates")) {
            in_atoms_coords = 1;
            continue;
        }
        if (in_atoms_coords && strstr(line, "Atoms.SpeciesAndCoordinates>")) {
            in_atoms_coords = 0;
            atoms_coords_read = sys->atom_count;
            continue;
        }
        if (in_atoms_coords) {
            int idx;
            char species_name[MAX_NAME_LEN];
            double cx, cy, cz;
            species_name[0] = '\0';
            int n = sscanf(line, "%d %63s %lf %lf %lf", &idx, species_name, &cx, &cy, &cz);
            if (n >= 5 && species_name[0] != '\0' && sys->atom_count < MAX_ATOMS) {
                char basis_name[MAX_NAME_LEN] = "";
                for (int d = 0; d < def_count; d++) {
                    if (strcmp(definitions[d].elem_name, species_name) == 0) {
                        strncpy(basis_name, definitions[d].basis_name, MAX_NAME_LEN - 1);
                        break;
                    }
                }
                int sp = resolve_species(sys, species_name, basis_name);
                if (sp < 0) { fclose(fp); return -1; }
                sys->atoms[sys->atom_count].species_idx = sp;
                parse_zeta_counts(basis_name, sys->atoms[sys->atom_count].zeta);

                if (strcmp(coord_unit, "Ang") == 0) {
                    sys->atoms[sys->atom_count].x = cx / BOHR_TO_ANGSTROM;
                    sys->atoms[sys->atom_count].y = cy / BOHR_TO_ANGSTROM;
                    sys->atoms[sys->atom_count].z = cz / BOHR_TO_ANGSTROM;
                } else if (strcmp(coord_unit, "AU") == 0) {
                    sys->atoms[sys->atom_count].x = cx;
                    sys->atoms[sys->atom_count].y = cy;
                    sys->atoms[sys->atom_count].z = cz;
                } else {
                    sys->atoms[sys->atom_count].x = cx;
                    sys->atoms[sys->atom_count].y = cy;
                    sys->atoms[sys->atom_count].z = cz;
                }
                sys->atom_count++;
            }
            continue;
        }


        if (strncmp(line, "<Atoms.UnitVectors", 18) == 0) {
            in_unit_vectors = 1;
            unit_vectors_read = 0;
            continue;
        }
        if (in_unit_vectors && strstr(line, "Atoms.UnitVectors>")) {
            in_unit_vectors = 0;
            continue;
        }
        if (in_unit_vectors && unit_vectors_read < 3) {
            double scale, tx, ty, tz;
            int n = sscanf(line, "%lf %lf %lf %lf", &scale, &tx, &ty, &tz);
            if (n >= 4) {
                unit_vectors_read++;
                tv_ang[unit_vectors_read][1] = tx * scale;
                tv_ang[unit_vectors_read][2] = ty * scale;
                tv_ang[unit_vectors_read][3] = tz * scale;
            } else if (n == 3) {
                unit_vectors_read++;
                tv_ang[unit_vectors_read][1] = scale;
                tv_ang[unit_vectors_read][2] = tx;
                tv_ang[unit_vectors_read][3] = ty;
            }
            continue;
        }
    }

    fflush(stderr); fclose(fp);

    for (int i = 1; i <= 3; i++) {
        sys->tv[i][1] = tv_ang[i][1] / BOHR_TO_ANGSTROM;
        sys->tv[i][2] = tv_ang[i][2] / BOHR_TO_ANGSTROM;
        sys->tv[i][3] = tv_ang[i][3] / BOHR_TO_ANGSTROM;
    }

    if (strcmp(coord_unit, "FRAC") == 0) {
        for (int a = 0; a < sys->atom_count; a++) {
            double fx = sys->atoms[a].x;
            double fy = sys->atoms[a].y;
            double fz = sys->atoms[a].z;
            sys->atoms[a].x = fx * sys->tv[1][1] + fy * sys->tv[2][1] + fz * sys->tv[3][1];
            sys->atoms[a].y = fx * sys->tv[1][2] + fy * sys->tv[2][2] + fz * sys->tv[3][2];
            sys->atoms[a].z = fx * sys->tv[1][3] + fy * sys->tv[2][3] + fz * sys->tv[3][3];
        }
    }

    for (int a = 0; a < sys->atom_count; a++) {
        int sp = sys->atoms[a].species_idx;
        int zeta_norbs = count_norbs_from_zeta(sys->atoms[a].zeta, sys->species[sp].Lmax);
        int pao_norbs = count_norbs_for_species(&sys->species[sp]);
        sys->atoms[a].norbs = (zeta_norbs > 0 && zeta_norbs < pao_norbs) ? zeta_norbs : pao_norbs;
    }

    if (sys->Ngrid1 != 0 && sys->Ngrid2 != 0 && sys->Ngrid3 != 0) {
        sys->has_fixed_grid = 1;
    } else if (sys->energycutoff > 0.0) {
        calc_grid_from_ecutoff(sys->tv, sys->energycutoff,
                              &sys->Ngrid1, &sys->Ngrid2, &sys->Ngrid3);
    } else {
        fprintf(stderr, "Error: Neither scf.Ngrid nor scf.energycutoff found\n");
        return -1;
    }

    if (sys->atom_count == 0) {
        fprintf(stderr, "Error: No atoms found in dat file\n");
        return -1;
    }

    return 0;
}

// ============================================================
// Wavefunction: binary loader + orbital index helpers
// ============================================================

typedef struct {
    double kx, ky, kz;
    int norbs;
    double _Complex *coeffs;
} Wavefunction;

static int count_norbs_total(const System *sys) {
    int n = 0;
    for (int a = 0; a < sys->atom_count; a++) {
        n += sys->atoms[a].norbs;
    }
    return n;
}

static int load_wavefunction(const char *wfn_path, System *sys, Wavefunction *wfn) {
    FILE *fp = fopen(wfn_path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open wavefunction file: %s\n", wfn_path);
        return -1;
    }

    double kp[3];
    if (fread(kp, sizeof(double), 3, fp) != 3) {
        fprintf(stderr, "Error: Failed to read k-point from %s\n", wfn_path);
        fflush(stderr); fclose(fp);
        return -1;
    }

    int total_norbs = count_norbs_total(sys);
    if (total_norbs <= 0) {
        fprintf(stderr, "Error: No orbitals in system (total_norbs=%d)\n", total_norbs);
        fflush(stderr); fclose(fp);
        return -1;
    }

    wfn->coeffs = (double _Complex *)calloc(total_norbs, sizeof(double _Complex));
    if (!wfn->coeffs) {
        fprintf(stderr, "Error: Cannot allocate memory for %d coefficients\n", total_norbs);
        fflush(stderr); fclose(fp);
        return -1;
    }

    for (int i = 0; i < total_norbs; i++) {
        double re, im;
        if (fread(&re, sizeof(double), 1, fp) != 1 ||
            fread(&im, sizeof(double), 1, fp) != 1) {
            fprintf(stderr, "Error: Failed to read coefficient %d from %s\n", i, wfn_path);
            free(wfn->coeffs);
            wfn->coeffs = NULL;
            fflush(stderr); fclose(fp);
            return -1;
        }
        wfn->coeffs[i] = re + im * I;
    }

    fflush(stderr); fclose(fp);

    wfn->kx = kp[0];
    wfn->ky = kp[1];
    wfn->kz = kp[2];
    wfn->norbs = total_norbs;

    printf("Wavefunction loaded: k=(%.4f, %.4f, %.4f), %d orbitals\n",
           wfn->kx, wfn->ky, wfn->kz, wfn->norbs);
    return 0;
}

static void free_wavefunction(Wavefunction *wfn) {
    if (wfn->coeffs) {
        free(wfn->coeffs);
        wfn->coeffs = NULL;
    }
    wfn->norbs = 0;
}

static int orbital_index(const System *sys, int atom_idx, int L, int Mul, int M) {
    if (atom_idx < 0 || atom_idx >= sys->atom_count) return -1;

    int sp = sys->atoms[atom_idx].species_idx;
    const SpeciesPAO *sps = &sys->species[sp];

    if (L < 0 || L > sps->Lmax || Mul < 0 || Mul >= sps->Mul) return -1;

    int m_count;
    if (L == 0) m_count = 1;
    else if (L == 1) m_count = 3;
    else if (L == 2) m_count = 5;
    else if (L == 3) m_count = 7;
    else m_count = 2 * L + 1;

    if (M < 0 || M >= m_count) return -1;

    int offset = 0;
    for (int a = 0; a < atom_idx; a++) {
        offset += sys->atoms[a].norbs;
    }
    for (int l = 0; l < L; l++) {
        int mc;
        if (l == 0) mc = 1;
        else if (l == 1) mc = 3;
        else if (l == 2) mc = 5;
        else if (l == 3) mc = 7;
        else mc = 2 * l + 1;
        offset += mc * sps->Mul;
    }
    offset += Mul * m_count;
    offset += M;

    return offset;
}

static int count_norbs_for_species_sys(const System *sys, int species_idx) {
    const SpeciesPAO *sp = &sys->species[species_idx];
    return count_norbs_for_species(sp);
}


static double eval_basis(const System *sys, int atom_idx,
                         int L, int Mul, int M,
                         double gx, double gy, double gz)
{
    const Atom *a = &sys->atoms[atom_idx];
    int sp = a->species_idx;
    const SpeciesPAO *sps = &sys->species[sp];

    double dx = gx - a->x;
    double dy = gy - a->y;
    double dz = gz - a->z;

    double R, theta, phi;
    xyz2spherical(dx, dy, dz, &R, &theta, &phi);

    double radial = PhiF(R, sps->RWF[L][Mul], sps->RV, sps->Mesh);
    double angular = AngularF(L, M, theta, phi);
    return radial * angular;
}

typedef struct {
    int atom_idx;
    int L;
    int Mul;
    int M;
} OrbitalInfo;

static void compute_wavefunction(const System *sys, const Wavefunction *wfn,
                                 double **out_real, double **out_imag, double **out_abs)
{
    int nx = sys->Ngrid1 + 1;
    int ny = sys->Ngrid2 + 1;
    int nz = sys->Ngrid3 + 1;
    size_t total = (size_t)nx * ny * nz;

    double *psi_r_arr = (double *)calloc(total, sizeof(double));
    double *psi_i_arr = (double *)calloc(total, sizeof(double));
    double *psi_abs = (double *)calloc(total, sizeof(double));
    if (!psi_r_arr || !psi_i_arr || !psi_abs) {
        free(psi_r_arr); free(psi_i_arr); free(psi_abs);
        *out_real = NULL; *out_imag = NULL; *out_abs = NULL;
        return;
    }

    double step_x[4], step_y[4], step_z[4];
    for (int c = 1; c <= 3; c++) {
        step_x[c] = sys->tv[c][1] / sys->Ngrid1;
        step_y[c] = sys->tv[c][2] / sys->Ngrid2;
        step_z[c] = sys->tv[c][3] / sys->Ngrid3;
    }

    int norbs = wfn->norbs;
    OrbitalInfo *orb_map = (OrbitalInfo *)malloc(norbs * sizeof(OrbitalInfo));
    if (!orb_map) {
        free(psi_r_arr); free(psi_i_arr); free(psi_abs);
        *out_real = NULL; *out_imag = NULL; *out_abs = NULL;
        return;
    }

    int o = 0;
    for (int a = 0; a < sys->atom_count; a++) {
        int sp = sys->atoms[a].species_idx;
        const SpeciesPAO *sps = &sys->species[sp];
        const int *z = sys->atoms[a].zeta;
        int Lmax = sps->Lmax;
        for (int L = 0; L <= Lmax; L++) {
            int m_count;
            if (L == 0) m_count = 1;
            else if (L == 1) m_count = 3;
            else if (L == 2) m_count = 5;
            else if (L == 3) m_count = 7;
            else m_count = 2 * L + 1;
            int nzeta = (L <= MAX_L && z[L] > 0) ? z[L] : 0;
            if (nzeta == 0) continue;
            for (int mul = 0; mul < nzeta; mul++) {
                for (int M = 0; M < m_count; M++) {
                    orb_map[o].atom_idx = a;
                    orb_map[o].L = L;
                    orb_map[o].Mul = mul;
                    orb_map[o].M = M;
                    o++;
                }
            }
        }
    }

    for (int i = 0; i < nx; i++) {
        if (i % 10 == 0) {
            fprintf(stderr, "Compute grid: %d/%d rows\n", i, nx);
        }
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                double gx = i * step_x[1] + j * step_x[2] + k * step_x[3];
                double gy = i * step_y[1] + j * step_y[2] + k * step_y[3];
                double gz = i * step_z[1] + j * step_z[2] + k * step_z[3];

                double psi_r = 0.0;
                double psi_i = 0.0;
                for (int o = 0; o < norbs; o++) {
                    double bv = eval_basis(sys, orb_map[o].atom_idx,
                                           orb_map[o].L, orb_map[o].Mul,
                                           orb_map[o].M, gx, gy, gz);
                    psi_r += creal(wfn->coeffs[o]) * bv;
                    psi_i += cimag(wfn->coeffs[o]) * bv;
                }
                size_t idx = (size_t)i * ny * nz + j * nz + k;
                psi_r_arr[idx] = psi_r;
                psi_i_arr[idx] = psi_i;
                psi_abs[idx] = psi_r * psi_r + psi_i * psi_i;
            }
        }
    }

    fprintf(stderr, "Compute grid: %d/%d rows\n", nx, nx);
    free(orb_map);

    *out_real = psi_r_arr;
    *out_imag = psi_i_arr;
    *out_abs = psi_abs;
}

// ============================================================
// Cube file writer
// ============================================================

static int species_name_to_Z(const char *name) {
    struct { const char *sym; int Z; } table[] = {
        {"H", 1}, {"He", 2}, {"Li", 3}, {"Be", 4}, {"B", 5},
        {"C", 6}, {"N", 7}, {"O", 8}, {"F", 9}, {"Ne", 10},
        {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"Si", 14}, {"P", 15},
        {"S", 16}, {"Cl", 17}, {"K", 19}, {"Ca", 20},
        {"Sc", 21}, {"Ti", 22}, {"V", 23}, {"Cr", 24}, {"Mn", 25},
        {"Fe", 26}, {"Co", 27}, {"Ni", 28}, {"Cu", 29}, {"Zn", 30},
        {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34}, {"Br", 35},
        {"Kr", 36}, {"Rb", 37}, {"Sr", 38}, {"Y", 39}, {"Zr", 40},
        {"Nb", 41}, {"Mo", 42}, {"Ag", 47}, {"Cd", 48}, {"In", 49},
        {"Sn", 50}, {"Sb", 51}, {"I", 53}, {"Xe", 54},
        {"Au", 79}, {"Hg", 80}, {"Pb", 82}, {"Bi", 83},
        {NULL, 0}
    };

    char sym[4] = {0};
    sym[0] = name[0];
    if (name[0] >= 'a' && name[0] <= 'z') {
        sym[0] = name[0] - 'a' + 'A';
    }
    if (name[1] >= 'a' && name[1] <= 'z') {
        sym[1] = name[1];
        sym[2] = '\0';
    } else {
        sym[1] = '\0';
    }

    for (int i = 0; table[i].sym != NULL; i++) {
        if (strcmp(sym, table[i].sym) == 0) {
            return table[i].Z;
        }
    }
    return 1;
}

static int write_cube(const char *out_path, const System *sys, const double *data, const char *description)
{
    FILE *fp = fopen(out_path, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open cube file for writing: %s\n", out_path);
        return -1;
    }

    int nx = sys->Ngrid1 + 1;
    int ny = sys->Ngrid2 + 1;
    int nz = sys->Ngrid3 + 1;

    fprintf(fp, "%s\n", description);
    fprintf(fp, "Generated by wfn2cube (HamGNN/DFT_interfaces/openmx/wfn_plot)\n");

    fprintf(fp, "%5d%11.6f%11.6f%11.6f\n", sys->atom_count, 0.0, 0.0, 0.0);

    double dx[4], dy[4], dz[4];
    for (int c = 1; c <= 3; c++) {
        dx[c] = sys->tv[c][1] / sys->Ngrid1;
        dy[c] = sys->tv[c][2] / sys->Ngrid2;
        dz[c] = sys->tv[c][3] / sys->Ngrid3;
    }
    fprintf(fp, "%5d%11.6f%11.6f%11.6f\n", nx, dx[1], dy[1], dz[1]);
    fprintf(fp, "%5d%11.6f%11.6f%11.6f\n", ny, dx[2], dy[2], dz[2]);
    fprintf(fp, "%5d%11.6f%11.6f%11.6f\n", nz, dx[3], dy[3], dz[3]);

    for (int a = 0; a < sys->atom_count; a++) {
        int sp = sys->atoms[a].species_idx;
        const char *sname = sys->species[sp].name;
        int Z = species_name_to_Z(sname);
        fprintf(fp, "%5d%11.6f%11.6f%11.6f%11.6f\n",
                Z, 0.0,
                sys->atoms[a].x,
                sys->atoms[a].y,
                sys->atoms[a].z);
    }

    int count = 0;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                double val = data[i * ny * nz + j * nz + k];
                fprintf(fp, "%13.5E", val);
                count++;
                if (count % 6 == 0) {
                    fprintf(fp, "\n");
                }
            }
        }
    }
    if (count % 6 != 0) {
        fprintf(fp, "\n");
    }

    fflush(stderr); fclose(fp);
    printf("Cube file written: %s (%d grid points)\n", out_path, count);
    return 0;
}

// ============================================================
// Test harness
// ============================================================

#ifdef TEST_DAT

#include <assert.h>
#include <sys/stat.h>

static void test_parse_dat_ang(void) {
    system("mkdir -p test_dat_ang_data/PAO");

    FILE *fp = fopen("test_dat_ang_data/PAO/He4.0.pao", "w");
    assert(fp);
    fprintf(fp, "PAO.Lmax  0\n");
    fprintf(fp, "PAO.Mul   1\n");
    fprintf(fp, "grid.num.output  10\n");
    fprintf(fp, "<pseudo.atomic.orbitals.L=0\n");
    for (int i = 0; i < 10; i++) {
        double rv = 0.1 * (i+1);
        fprintf(fp, "%.6f  %.6f  %.10f\n", rv*1.8897, rv, exp(-rv));
    }
    fprintf(fp, "pseudo.atomic.orbitals.L=0>\n");
    fflush(stderr); fclose(fp);

    fp = fopen("test_dat_ang_data/PAO/H4.0.pao", "w");
    assert(fp);
    fprintf(fp, "PAO.Lmax  1\n");
    fprintf(fp, "PAO.Mul   1\n");
    fprintf(fp, "grid.num.output  10\n");
    for (int L = 0; L <= 1; L++) {
        fprintf(fp, "<pseudo.atomic.orbitals.L=%d\n", L);
        for (int i = 0; i < 10; i++) {
            double rv = 0.1 * (i+1);
            fprintf(fp, "%.6f  %.6f  %.10f\n", rv*1.8897, rv, exp(-rv));
        }
        fprintf(fp, "pseudo.atomic.orbitals.L=%d>\n", L);
    }
    fflush(stderr); fclose(fp);

    fp = fopen("test_dat_ang.dat", "w");
    assert(fp);
    fprintf(fp, "DATA.PATH           ./test_dat_ang_data\n");
    fprintf(fp, "<Definition.of.Atomic.Species\n");
    fprintf(fp, "1  He  He4.0-HOCP  He4.0-HOCP  4.0026\n");
    fprintf(fp, "2  H   H4.0-s2p1   H4.0-s2p1   1.008\n");
    fprintf(fp, "Definition.of.Atomic.Species>\n");
    fprintf(fp, "scf.energycutoff    150.0\n");
    fprintf(fp, "Atoms.SpeciesAndCoordinates.Unit  Ang\n");
    fprintf(fp, "<Atoms.SpeciesAndCoordinates\n");
    fprintf(fp, "1  He  0.0  0.0  0.0\n");
    fprintf(fp, "2  H   2.0  2.0  2.0\n");
    fprintf(fp, "Atoms.SpeciesAndCoordinates>\n");
    fprintf(fp, "<Atoms.UnitVectors\n");
    fprintf(fp, "1.0  4.0  0.0  0.0\n");
    fprintf(fp, "1.0  0.0  4.0  0.0\n");
    fprintf(fp, "1.0  0.0  0.0  4.0\n");
    fprintf(fp, "Atoms.UnitVectors>\n");
    fflush(stderr); fclose(fp);

    System sys;
    int ret = parse_dat("test_dat_ang.dat", &sys);
    assert(ret == 0);
    assert(sys.atom_count == 2);
    assert(sys.species_count == 2);
    assert(sys.energycutoff == 150.0);
    assert(strcmp(sys.data_path, "./test_dat_ang_data") == 0);

    double a = M_PI / sqrt(150.0);
    int expected_n = (int)ceil((4.0 / BOHR_TO_ANGSTROM) / a);
    while (!is_fft_friendly(expected_n)) expected_n++;
    assert(sys.Ngrid1 == expected_n && sys.Ngrid2 == expected_n && sys.Ngrid3 == expected_n);

    double expected_lattice = 4.0 / BOHR_TO_ANGSTROM;
    assert(fabs(sys.tv[1][1] - expected_lattice) < 1e-6);
    assert(fabs(sys.tv[2][2] - expected_lattice) < 1e-6);
    assert(fabs(sys.tv[3][3] - expected_lattice) < 1e-6);

    assert(fabs(sys.atoms[0].x) < 1e-10);
    assert(fabs(sys.atoms[0].y) < 1e-10);
    assert(fabs(sys.atoms[0].z) < 1e-10);

    double expected_coord = 2.0 / BOHR_TO_ANGSTROM;
    assert(fabs(sys.atoms[1].x - expected_coord) < 1e-6);
    assert(fabs(sys.atoms[1].y - expected_coord) < 1e-6);
    assert(fabs(sys.atoms[1].z - expected_coord) < 1e-6);

    assert(sys.atoms[0].norbs == 1);
    assert(sys.atoms[1].norbs == 4);

    printf("test_parse_dat_ang: PASS (grid=%d from ecutoff=150)\n", expected_n);

    remove("test_dat_ang.dat");
    remove("test_dat_ang_data/PAO/He4.0.pao");
    remove("test_dat_ang_data/PAO/H4.0.pao");
    rmdir("test_dat_ang_data/PAO");
    rmdir("test_dat_ang_data");
    free_system(&sys);
}

static void test_parse_dat_frac(void) {
    system("mkdir -p test_dat_frac_data/PAO");

    FILE *fp = fopen("test_dat_frac_data/PAO/Si5.0.pao", "w");
    assert(fp);
    fprintf(fp, "PAO.Lmax  1\n");
    fprintf(fp, "PAO.Mul   2\n");
    fprintf(fp, "grid.num.output  15\n");
    for (int L = 0; L <= 1; L++) {
        fprintf(fp, "<pseudo.atomic.orbitals.L=%d\n", L);
        for (int i = 0; i < 15; i++) {
            double rv = 0.1 * (i+1);
            fprintf(fp, "%.6f  %.6f  %.10f  %.10f\n", rv*1.8897, rv, exp(-rv), 0.5*exp(-rv));
        }
        fprintf(fp, "pseudo.atomic.orbitals.L=%d>\n", L);
    }
    fflush(stderr); fclose(fp);

    fp = fopen("test_dat_frac.dat", "w");
    assert(fp);
    fprintf(fp, "DATA.PATH           ./test_dat_frac_data\n");
    fprintf(fp, "<Definition.of.Atomic.Species\n");
    fprintf(fp, "1  Si  Si5.0-s2p2d1  Si5.0-s2p2d1  28.0855\n");
    fprintf(fp, "Definition.of.Atomic.Species>\n");
    fprintf(fp, "scf.energycutoff    200.0\n");
    fprintf(fp, "Atoms.SpeciesAndCoordinates.Unit  FRAC\n");
    fprintf(fp, "<Atoms.SpeciesAndCoordinates\n");
    fprintf(fp, "1  Si  0.0  0.0  0.0\n");
    fprintf(fp, "2  Si  0.25 0.25 0.25\n");
    fprintf(fp, "Atoms.SpeciesAndCoordinates>\n");
    fprintf(fp, "<Atoms.UnitVectors\n");
    fprintf(fp, "1.0  5.43  0.0   0.0\n");
    fprintf(fp, "1.0  0.0   5.43  0.0\n");
    fprintf(fp, "1.0  0.0   0.0   5.43\n");
    fprintf(fp, "Atoms.UnitVectors>\n");
    fflush(stderr); fclose(fp);

    System sys;
    int ret = parse_dat("test_dat_frac.dat", &sys);
    assert(ret == 0);
    assert(sys.atom_count == 2);
    assert(sys.species_count == 1);
    assert(sys.energycutoff == 200.0);

    double lattice_bohr = 5.43 / BOHR_TO_ANGSTROM;

    assert(fabs(sys.atoms[0].x) < 1e-10);
    assert(fabs(sys.atoms[0].y) < 1e-10);
    assert(fabs(sys.atoms[0].z) < 1e-10);

    double expected = 0.25 * lattice_bohr;
    assert(fabs(sys.atoms[1].x - expected) < 1e-6);
    assert(fabs(sys.atoms[1].y - expected) < 1e-6);
    assert(fabs(sys.atoms[1].z - expected) < 1e-6);

    assert(sys.atoms[0].norbs == 8);
    assert(sys.atoms[1].norbs == 8);

    double a = M_PI / sqrt(200.0);
    int expected_n = (int)ceil(lattice_bohr / a);
    while (!is_fft_friendly(expected_n)) expected_n++;
    assert(sys.Ngrid1 == expected_n && sys.Ngrid2 == expected_n && sys.Ngrid3 == expected_n);

    printf("test_parse_dat_frac: PASS (grid=%d from ecutoff=200)\n", expected_n);

    remove("test_dat_frac.dat");
    remove("test_dat_frac_data/PAO/Si5.0.pao");
    rmdir("test_dat_frac_data/PAO");
    rmdir("test_dat_frac_data");
    free_system(&sys);
}

static void test_parse_dat_noindex(void) {
    system("mkdir -p test_dat_noindex_data/PAO");

    FILE *fp = fopen("test_dat_noindex_data/PAO/Si6.0.pao", "w");
    assert(fp);
    fprintf(fp, "PAO.Lmax  1\n");
    fprintf(fp, "PAO.Mul   2\n");
    fprintf(fp, "grid.num.output  10\n");
    for (int L = 0; L <= 1; L++) {
        fprintf(fp, "<pseudo.atomic.orbitals.L=%d\n", L);
        for (int i = 0; i < 10; i++) {
            double rv = 0.1 * (i+1);
            fprintf(fp, "%.6f  %.6f  %.10f  %.10f\n", rv*1.8897, rv, exp(-rv), 0.5*exp(-rv));
        }
        fprintf(fp, "pseudo.atomic.orbitals.L=%d>\n", L);
    }
    fflush(stderr); fclose(fp);

    fp = fopen("test_dat_noindex.dat", "w");
    assert(fp);
    fprintf(fp, "DATA.PATH           ./test_dat_noindex_data\n");
    fprintf(fp, "scf.energycutoff    150.0\n");
    fprintf(fp, "<Definition.of.Atomic.Species\n");
    fprintf(fp, "Si   Si6.0-s2p2       Si_PBE19\n");
    fprintf(fp, "Definition.of.Atomic.Species>\n");
    fprintf(fp, "Atoms.SpeciesAndCoordinates.Unit  Ang\n");
    fprintf(fp, " <Atoms.SpeciesAndCoordinates\n");
    fprintf(fp, "  1  Si   0.0   0.0   0.0   4.0   4.0\n");
    fprintf(fp, "  2  Si   1.0   1.0   1.0   4.0   4.0\n");
    fprintf(fp, "Atoms.SpeciesAndCoordinates>\n");
    fprintf(fp, "<Atoms.UnitVectors\n");
    fprintf(fp, "1.0  5.0  0.0  0.0\n");
    fprintf(fp, "1.0  0.0  5.0  0.0\n");
    fprintf(fp, "1.0  0.0  0.0  5.0\n");
    fprintf(fp, "Atoms.UnitVectors>\n");
    fflush(stderr); fclose(fp);

    System sys;
    int ret = parse_dat("test_dat_noindex.dat", &sys);
    assert(ret == 0);
    assert(sys.atom_count == 2);
    assert(sys.species_count == 1);
    assert(strcmp(sys.species[0].basis_name, "Si6.0") == 0);
    assert(strcmp(sys.species[0].elem_name, "Si") == 0);

    printf("test_parse_dat_noindex: PASS\n");

    remove("test_dat_noindex.dat");
    remove("test_dat_noindex_data/PAO/Si6.0.pao");
    rmdir("test_dat_noindex_data/PAO");
    rmdir("test_dat_noindex_data");
    free_system(&sys);
}

static void test_parse_dat_fixed_grid(void) {
    system("mkdir -p test_dat_fixed_data/PAO");

    FILE *fp = fopen("test_dat_fixed_data/PAO/He4.0.pao", "w");
    assert(fp);
    fprintf(fp, "PAO.Lmax  0\n");
    fprintf(fp, "PAO.Mul   1\n");
    fprintf(fp, "grid.num.output  10\n");
    fprintf(fp, "<pseudo.atomic.orbitals.L=0\n");
    for (int i = 0; i < 10; i++) {
        double rv = 0.1 * (i+1);
        fprintf(fp, "%.6f  %.6f  %.10f\n", rv*1.8897, rv, exp(-rv));
    }
    fprintf(fp, "pseudo.atomic.orbitals.L=0>\n");
    fflush(stderr); fclose(fp);

    fp = fopen("test_dat_fixed.dat", "w");
    assert(fp);
    fprintf(fp, "DATA.PATH           ./test_dat_fixed_data\n");
    fprintf(fp, "<Definition.of.Atomic.Species\n");
    fprintf(fp, "1  He  He4.0-HOCP  He4.0-HOCP  4.0026\n");
    fprintf(fp, "Definition.of.Atomic.Species>\n");
    fprintf(fp, "scf.Ngrid1           32\n");
    fprintf(fp, "scf.Ngrid2           48\n");
    fprintf(fp, "scf.Ngrid3           64\n");
    fprintf(fp, "scf.energycutoff    150.0\n");
    fprintf(fp, "Atoms.SpeciesAndCoordinates.Unit  Ang\n");
    fprintf(fp, "<Atoms.SpeciesAndCoordinates\n");
    fprintf(fp, "1  He  0.0  0.0  0.0\n");
    fprintf(fp, "Atoms.SpeciesAndCoordinates>\n");
    fprintf(fp, "<Atoms.UnitVectors\n");
    fprintf(fp, "1.0  4.0  0.0  0.0\n");
    fprintf(fp, "1.0  0.0  4.0  0.0\n");
    fprintf(fp, "1.0  0.0  0.0  4.0\n");
    fprintf(fp, "Atoms.UnitVectors>\n");
    fflush(stderr); fclose(fp);

    System sys;
    int ret = parse_dat("test_dat_fixed.dat", &sys);
    assert(ret == 0);
    assert(sys.has_fixed_grid == 1);
    assert(sys.Ngrid1 == 32 && sys.Ngrid2 == 48 && sys.Ngrid3 == 64);

    printf("test_parse_dat_fixed_grid: PASS (fixed grid 32x48x64)\n");

    remove("test_dat_fixed.dat");
    remove("test_dat_fixed_data/PAO/He4.0.pao");
    rmdir("test_dat_fixed_data/PAO");
    rmdir("test_dat_fixed_data");
    free_system(&sys);
}

static void test_calc_grid_from_ecutoff(void) {
    double tv[4][4];
    double lattice_bohr = 10.0;
    for (int i = 1; i <= 3; i++) {
        tv[i][i] = lattice_bohr;
        for (int j = 1; j <= 3; j++) {
            if (i != j) tv[i][j] = 0;
        }
    }

    int N1, N2, N3;
    calc_grid_from_ecutoff(tv, 150.0, &N1, &N2, &N3);

    double a = M_PI / sqrt(150.0);
    int expected_n = (int)ceil(lattice_bohr / a);
    while (!is_fft_friendly(expected_n)) expected_n++;

    assert(N1 == expected_n && N2 == expected_n && N3 == expected_n);
    assert(is_fft_friendly(N1));

    printf("test_calc_grid_from_ecutoff: PASS (n=%d for a=%.4f)\n", N1, a);
}

static void test_parse_dat_missing_file(void) {
    System sys;
    int ret = parse_dat("/nonexistent/openmx_missing.dat", &sys);
    assert(ret == -1);
    printf("test_parse_dat_missing_file: PASS\n");
}

int main(void) {
    test_parse_dat_ang();
    test_parse_dat_frac();
    test_parse_dat_noindex();
    test_parse_dat_fixed_grid();
    test_calc_grid_from_ecutoff();
    test_parse_dat_missing_file();
    printf("\nAll DAT parser tests PASSED\n");
    return 0;
}
#endif

#ifdef TEST_MATH

#include <assert.h>
#include <float.h>

static void test_xyz2spherical(void) {
    double R, theta, phi;

    xyz2spherical(0, 0, 1.0, &R, &theta, &phi);
    assert(R > 0.999 && R < 1.001);
    assert(theta < 0.001);

    xyz2spherical(1.0, 0, 0, &R, &theta, &phi);
    assert(fabs(theta - M_PI/2) < 0.001);
    assert(fabs(phi) < 0.001);

    xyz2spherical(0, 1.0, 0, &R, &theta, &phi);
    assert(fabs(theta - M_PI/2) < 0.001);
    assert(fabs(phi - M_PI/2) < 0.001);

    xyz2spherical(0, 0, 0, &R, &theta, &phi);
    assert(R < 1e-13);

    printf("test_xyz2spherical: PASS\n");
}

static void test_AngularF(void) {
    double expected_s = 0.5 / sqrt(M_PI);
    assert(fabs(AngularF(0, 0, 0.1, 0.2) - expected_s) < 1e-10);

    double Q = 0.0, P = 0.0;
    double expected_pz = 0.5 * sqrt(3.0/M_PI);
    assert(fabs(AngularF(1, 2, Q, P) - expected_pz) < 1e-10);

    Q = M_PI/2; P = 0.0;
    double expected_px = 0.5 * sqrt(3.0/M_PI);
    assert(fabs(AngularF(1, 0, Q, P) - expected_px) < 1e-10);

    Q = M_PI/2; P = M_PI/2;
    double expected_py = 0.5 * sqrt(3.0/M_PI);
    assert(fabs(AngularF(1, 1, Q, P) - expected_py) < 1e-10);

    printf("test_AngularF: PASS\n");
}

static void test_PhiF(void) {
    double RV[] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    double phi0[] = {1.0, 0.9, 0.6, 0.3, 0.1, 0.03, 0.01};
    int N = 7;

    double val = PhiF(1.0, phi0, RV, N);
    assert(val > 0.5 && val < 0.7);

    // Out of range above cutoff
    assert(PhiF(5.0, phi0, RV, N) == 0.0);

    // Near origin
    val = PhiF(0.01, phi0, RV, N);
    assert(val > 0.8);

    // Upper boundary (regression test for OOB buffer overflow fix)
    // Hermite spline may slightly overshoot; just verify no crash and small magnitude
    val = PhiF(2.9, phi0, RV, N);
    assert(fabs(val) < 0.2);

    // Below RV[0] - parabolic extrapolation from m=4 (matches PhiF.c)
    // Test uses uniform grid from 0; extrapolation from m=4 is far from origin.
    // Real PAO files have RV[0] ~ 0.001, where extrapolation is accurate.
    val = PhiF(-0.1, phi0, RV, N);
    assert(fabs(val - 0.369) < 0.01);

    printf("test_PhiF: PASS\n");
}

int main(void) {
    test_xyz2spherical();
    test_AngularF();
    test_PhiF();
    printf("\nAll math tests PASSED\n");
    return 0;
}
#endif

#ifdef TEST_PAO
#include <assert.h>
#include <sys/stat.h>

static void test_load_pao(void) {
    System sys;
    memset(&sys, 0, sizeof(sys));
    system("mkdir -p test_pao_data/PAO");

    FILE *fp = fopen("test_pao_data/PAO/He4.0.pao", "w");
    assert(fp);
    fprintf(fp, "PAO.Lmax  0\n");
    fprintf(fp, "PAO.Mul   1\n");
    fprintf(fp, "grid.num.output  20\n");
    fprintf(fp, "<pseudo.atomic.orbitals.L=0\n");
    for (int i = 0; i < 20; i++) {
        double rv = 0.05 * (i+1);
        fprintf(fp, "%.6f  %.6f  %.10f\n", rv*1.8897, rv, exp(-rv));
    }
    fprintf(fp, "pseudo.atomic.orbitals.L=0>\n");
    fflush(stderr); fclose(fp);

    strncpy(sys.data_path, "test_pao_data", sizeof(sys.data_path)-1);
    int idx = resolve_species(&sys, "He", "He4.0-HOCP");
    assert(idx == 0);
    assert(sys.species[0].Lmax == 0);
    assert(sys.species[0].Mul == 1);
    assert(sys.species[0].Mesh == 20);
    assert(sys.species[0].RV[0] > 0.0);
    double expected = exp(-0.05);
    assert(fabs(sys.species[0].RWF[0][0][0] - expected) < 1e-6);

    int idx2 = resolve_species(&sys, "He", "He4.0-HOCP");
    assert(idx2 == 0);
    assert(sys.species_count == 1);

    printf("test_load_pao: PASS\n");

    remove("test_pao_data/PAO/He4.0.pao");
    rmdir("test_pao_data/PAO");
    rmdir("test_pao_data");
    free_system(&sys);
}

static void test_load_pao_multiple_L(void) {
    System sys;
    memset(&sys, 0, sizeof(sys));
    system("mkdir -p test_pao_data2/PAO");

    FILE *fp = fopen("test_pao_data2/PAO/C5.0.pao", "w");
    assert(fp);
    fprintf(fp, "PAO.Lmax  2\n");
    fprintf(fp, "PAO.Mul   2\n");
    fprintf(fp, "grid.num.output  10\n");
    for (int L = 0; L <= 2; L++) {
        fprintf(fp, "<pseudo.atomic.orbitals.L=%d\n", L);
        for (int i = 0; i < 10; i++) {
            double rv = 0.1 * (i+1);
            fprintf(fp, "%.6f  %.6f  %.6f  %.6f\n", rv*1.8897, rv,
                    (1.0/(L+1))*exp(-rv), (1.0/(L+2))*exp(-rv));
        }
        fprintf(fp, "pseudo.atomic.orbitals.L=%d>\n", L);
    }
    fflush(stderr); fclose(fp);

    strncpy(sys.data_path, "test_pao_data2", sizeof(sys.data_path)-1);
    int idx = resolve_species(&sys, "C", "C5.0-s2p2d1");
    assert(idx == 0);
    assert(sys.species[0].Lmax == 2);
    assert(sys.species[0].Mul == 2);
    assert(sys.species[0].Mesh == 10);

    assert(sys.species[0].RWF[0][0][0] > 0.0);
    assert(sys.species[0].RWF[1][0][0] > 0.0);
    assert(sys.species[0].RWF[2][0][0] > 0.0);
    assert(sys.species[0].RWF[0][1][0] > 0.0);

    printf("test_load_pao_multiple_L: PASS\n");

    remove("test_pao_data2/PAO/C5.0.pao");
    rmdir("test_pao_data2/PAO");
    rmdir("test_pao_data2");
    free_system(&sys);
}

static void test_load_pao_nonexistent(void) {
    System sys;
    memset(&sys, 0, sizeof(sys));
    strncpy(sys.data_path, "/nonexistent/path", sizeof(sys.data_path)-1);
    int ret = resolve_species(&sys, "He", "He4.0-HOCP");
    assert(ret == -1);
    printf("test_load_pao_nonexistent: PASS\n");
    free_system(&sys);
}

int main(void) {
    test_load_pao();
    test_load_pao_multiple_L();
    test_load_pao_nonexistent();
    printf("\nAll PAO parser tests PASSED\n");
    return 0;
}
#endif

#ifdef TEST_WFN
#include <assert.h>
#include <math.h>

static void test_wfn_load_minimal(void) {
    FILE *fp = fopen("test_minimal.bin", "wb");
    assert(fp);
    double kp[3] = {0.0, 0.0, 0.0};
    fwrite(kp, sizeof(double), 3, fp);
    double c1_re = 1.0+0.0*I, c1_im = 0.0;
    double c1i_re = 0.0, c1i_im = 2.0;
    fwrite(&c1_re, sizeof(double), 1, fp);
    fwrite(&c1i_im, sizeof(double), 1, fp);
    double c2_re = 3.0, c2_im = 4.0;
    fwrite(&c2_re, sizeof(double), 1, fp);
    fwrite(&c2_im, sizeof(double), 1, fp);
    fflush(stderr); fclose(fp);

    System sys;
    memset(&sys, 0, sizeof(sys));
    sys.atom_count = 1;
    sys.atoms[0].norbs = 2;
    sys.atoms[0].species_idx = 0;
    sys.species[0].Lmax = 0;
    sys.species[0].Mul = 1;

    Wavefunction wfn;
    memset(&wfn, 0, sizeof(wfn));
    int ret = load_wavefunction("test_minimal.bin", &sys, &wfn);
    assert(ret == 0);
    assert(wfn.norbs == 2);
    assert(wfn.kx == 0.0 && wfn.ky == 0.0 && wfn.kz == 0.0);
    assert(fabs(creal(wfn.coeffs[0]) - 1.0) < 1e-10);
    assert(fabs(cimag(wfn.coeffs[0]) - 2.0) < 1e-10);
    assert(fabs(creal(wfn.coeffs[1]) - 3.0) < 1e-10);
    assert(fabs(cimag(wfn.coeffs[1]) - 4.0) < 1e-10);

    printf("test_wfn_load_minimal: PASS\n");
    free_wavefunction(&wfn);
    remove("test_minimal.bin");
}

static void test_wfn_load_4orbitals(void) {
    FILE *fp = fopen("test_4orbs.bin", "wb");
    assert(fp);
    double kp[3] = {0.5, 0.25, 0.0};
    fwrite(kp, sizeof(double), 3, fp);
    for (int i = 0; i < 4; i++) {
        double re = (i+1)*1.0;
        double im = (i+1)*0.5;
        fwrite(&re, sizeof(double), 1, fp);
        fwrite(&im, sizeof(double), 1, fp);
    }
    fflush(stderr); fclose(fp);

    System sys;
    memset(&sys, 0, sizeof(sys));
    sys.atom_count = 2;
    sys.atoms[0].norbs = 2;
    sys.atoms[0].species_idx = 0;
    sys.atoms[1].norbs = 2;
    sys.atoms[1].species_idx = 0;
    sys.species[0].Lmax = 0;
    sys.species[0].Mul = 1;

    Wavefunction wfn;
    memset(&wfn, 0, sizeof(wfn));
    int ret = load_wavefunction("test_4orbs.bin", &sys, &wfn);
    assert(ret == 0);
    assert(wfn.norbs == 4);
    assert(fabs(wfn.kx - 0.5) < 1e-10);
    assert(fabs(wfn.ky - 0.25) < 1e-10);
    assert(fabs(wfn.kz) < 1e-10);

    /* Verify coefficient ordering: first 2 belong to atom 0, next 2 to atom 1 */
    assert(fabs(creal(wfn.coeffs[0]) - 1.0) < 1e-10);
    assert(fabs(creal(wfn.coeffs[1]) - 2.0) < 1e-10);
    assert(fabs(creal(wfn.coeffs[2]) - 3.0) < 1e-10);
    assert(fabs(creal(wfn.coeffs[3]) - 4.0) < 1e-10);

    printf("test_wfn_load_4orbitals: PASS\n");
    free_wavefunction(&wfn);
    remove("test_4orbs.bin");
}

static void test_orbital_index(void) {
    System sys;
    memset(&sys, 0, sizeof(sys));
    sys.atom_count = 2;
    sys.species_count = 1;
    sys.atoms[0].species_idx = 0;
    sys.atoms[1].species_idx = 0;
    sys.species[0].Lmax = 1;
    sys.species[0].Mul = 1;

    /* Atom 0: L=0 has 1 orb; L=1 has 3 orbs -> norbs=4 */
    /* Atom 1: same -> norbs=4, global offset=4 */
    sys.atoms[0].norbs = 4;
    sys.atoms[1].norbs = 4;

    /* (atom=0, L=0, Mul=0, M=0) -> index 0 */
    assert(orbital_index(&sys, 0, 0, 0, 0) == 0);

    /* (atom=0, L=1, Mul=0, M=0) -> index 1 */
    assert(orbital_index(&sys, 0, 1, 0, 0) == 1);

    /* (atom=0, L=1, Mul=0, M=1) -> index 2 */
    assert(orbital_index(&sys, 0, 1, 0, 1) == 2);

    /* (atom=0, L=1, Mul=0, M=2) -> index 3 */
    assert(orbital_index(&sys, 0, 1, 0, 2) == 3);

    /* (atom=1, L=0, Mul=0, M=0) -> index 4 */
    assert(orbital_index(&sys, 1, 0, 0, 0) == 4);

    /* (atom=1, L=1, Mul=0, M=2) -> index 7 */
    assert(orbital_index(&sys, 1, 1, 0, 2) == 7);

    /* Out of range checks */
    assert(orbital_index(&sys, -1, 0, 0, 0) == -1);
    assert(orbital_index(&sys, 2, 0, 0, 0) == -1);
    assert(orbital_index(&sys, 0, 2, 0, 0) == -1);
    assert(orbital_index(&sys, 0, 1, 0, 3) == -1);

    printf("test_orbital_index: PASS\n");
}

static void test_wfn_missing_file(void) {
    System sys;
    memset(&sys, 0, sizeof(sys));
    Wavefunction wfn;
    memset(&wfn, 0, sizeof(wfn));
    int ret = load_wavefunction("/nonexistent/test_missing.bin", &sys, &wfn);
    assert(ret == -1);
    assert(wfn.coeffs == NULL);
    printf("test_wfn_missing_file: PASS\n");
}

int main(void) {
    test_wfn_load_minimal();
    test_wfn_load_4orbitals();
    test_orbital_index();
    test_wfn_missing_file();
    printf("\nAll WFN loader tests PASSED\n");
    return 0;
}
#endif

#ifdef TEST_COMPUTE
#include <assert.h>

static void test_compute_single_he(void) {
    System sys;
    memset(&sys, 0, sizeof(sys));

    sys.atom_count = 1;
    sys.atoms[0].species_idx = 0;
    sys.atoms[0].x = 0.0;
    sys.atoms[0].y = 0.0;
    sys.atoms[0].z = 0.0;

    sys.species_count = 1;
    sys.species[0].Lmax = 0;
    sys.species[0].Mul = 1;
    sys.species[0].Mesh = 100;
    sys.species[0].RV = (double *)calloc(100, sizeof(double));
    sys.species[0].RWF = (double ***)calloc(1, sizeof(double **));
    sys.species[0].RWF[0] = (double **)calloc(1, sizeof(double *));
    sys.species[0].RWF[0][0] = (double *)calloc(100, sizeof(double));

    double max_r = 8.0;
    for (int i = 0; i < 100; i++) {
        sys.species[0].RV[i] = max_r * i / 99.0;
        double r = sys.species[0].RV[i];
        sys.species[0].RWF[0][0][i] = exp(-r);
    }

    sys.Ngrid1 = 3;
    sys.Ngrid2 = 3;
    sys.Ngrid3 = 3;
    sys.tv[1][1] = 4.0; sys.tv[1][2] = 0.0; sys.tv[1][3] = 0.0;
    sys.tv[2][1] = 0.0; sys.tv[2][2] = 4.0; sys.tv[2][3] = 0.0;
    sys.tv[3][1] = 0.0; sys.tv[3][2] = 0.0; sys.tv[3][3] = 4.0;

    sys.atoms[0].norbs = 1;

    Wavefunction wfn;
    memset(&wfn, 0, sizeof(wfn));
    wfn.norbs = 1;
    wfn.coeffs = (double _Complex *)calloc(1, sizeof(double _Complex));
    wfn.coeffs[0] = 1.0 + 0.0 * I;

    double *psi_real = NULL, *psi_imag = NULL, *psi_abs = NULL;
    compute_wavefunction(&sys, &wfn, &psi_real, &psi_imag, &psi_abs);
    assert(psi_real != NULL);
    assert(psi_imag != NULL);
    assert(psi_abs != NULL);

    int nx = 4, ny = 4, nz = 4;
    double at_origin = psi_abs[0];
    assert(at_origin > 0.0);

    double at_corner = psi_abs[(nx-1)*ny*nz + (ny-1)*nz + (nz-1)];
    assert(at_corner < at_origin);

    printf("origin |psi|^2 = %.10f\n", at_origin);
    printf("corner |psi|^2 = %.10f\n", at_corner);
    printf("ratio = %.6f\n", at_corner / at_origin);
    printf("origin Re[psi] = %.10f\n", psi_real[0]);

    free(psi_real);
    free(psi_imag);
    free(psi_abs);
    free(wfn.coeffs);
    free_species_pao(&sys.species[0]);
}

int main(void) {
    test_compute_single_he();
    printf("\nAll COMPUTE tests PASSED\n");
    return 0;
}
#endif

#ifdef TEST_CUBE
#include <assert.h>
#include <float.h>

static void test_cube_writer(void) {
    System sys;
    memset(&sys, 0, sizeof(sys));

    sys.atom_count = 1;
    sys.atoms[0].species_idx = 0;
    sys.atoms[0].x = 0.0;
    sys.atoms[0].y = 0.0;
    sys.atoms[0].z = 0.0;

    sys.species_count = 1;
    sys.species[0].Lmax = 0;
    sys.species[0].Mul = 1;
    sys.species[0].Mesh = 100;
    sys.species[0].RV = (double *)calloc(100, sizeof(double));
    sys.species[0].RWF = (double ***)calloc(1, sizeof(double **));
    sys.species[0].RWF[0] = (double **)calloc(1, sizeof(double *));
    sys.species[0].RWF[0][0] = (double *)calloc(100, sizeof(double));
    strncpy(sys.species[0].name, "He", MAX_NAME_LEN - 1);

    double max_r = 8.0;
    for (int i = 0; i < 100; i++) {
        sys.species[0].RV[i] = max_r * i / 99.0;
        double r = sys.species[0].RV[i];
        sys.species[0].RWF[0][0][i] = exp(-r);
    }

    sys.Ngrid1 = 3;
    sys.Ngrid2 = 3;
    sys.Ngrid3 = 3;
    sys.tv[1][1] = 4.0; sys.tv[1][2] = 0.0; sys.tv[1][3] = 0.0;
    sys.tv[2][1] = 0.0; sys.tv[2][2] = 4.0; sys.tv[2][3] = 0.0;
    sys.tv[3][1] = 0.0; sys.tv[3][2] = 0.0; sys.tv[3][3] = 4.0;

    sys.atoms[0].norbs = 1;

    Wavefunction wfn;
    memset(&wfn, 0, sizeof(wfn));
    wfn.norbs = 1;
    wfn.coeffs = (double _Complex *)calloc(1, sizeof(double _Complex));
    wfn.coeffs[0] = 1.0 + 0.0 * I;

    double *psi_real = NULL, *psi_imag = NULL, *psi_abs = NULL;
    compute_wavefunction(&sys, &wfn, &psi_real, &psi_imag, &psi_abs);
    assert(psi_real != NULL);
    assert(psi_imag != NULL);
    assert(psi_abs != NULL);

    int ret = write_cube("test_output.cube", &sys, psi_abs, "Wavefunction probability density |psi|^2");
    assert(ret == 0);

    FILE *fp = fopen("test_output.cube", "r");
    assert(fp != NULL);

    char line[1024];

    assert(fgets(line, sizeof(line), fp) != NULL);
    assert(strstr(line, "psi") != NULL || strstr(line, "Wavefunction") != NULL);
    assert(fgets(line, sizeof(line), fp) != NULL);

    assert(fgets(line, sizeof(line), fp) != NULL);
    int natom;
    double ox, oy, oz;
    sscanf(line, "%d %lf %lf %lf", &natom, &ox, &oy, &oz);
    assert(natom == 1);
    assert(fabs(ox) < 1e-6 && fabs(oy) < 1e-6 && fabs(oz) < 1e-6);

    int nx_h, ny_h, nz_h;
    double sx1, sy1, sz1, sx2, sy2, sz2, sx3, sy3, sz3;
    assert(fgets(line, sizeof(line), fp) != NULL);
    sscanf(line, "%d %lf %lf %lf", &nx_h, &sx1, &sy1, &sz1);
    assert(nx_h == 4);
    assert(fabs(sx1 - 4.0/3.0) < 1e-6);
    assert(fabs(sy1) < 1e-6);
    assert(fabs(sz1) < 1e-6);

    assert(fgets(line, sizeof(line), fp) != NULL);
    sscanf(line, "%d %lf %lf %lf", &ny_h, &sx2, &sy2, &sz2);
    assert(ny_h == 4);
    assert(fabs(sx2) < 1e-6);
    assert(fabs(sy2 - 4.0/3.0) < 1e-6);

    assert(fgets(line, sizeof(line), fp) != NULL);
    sscanf(line, "%d %lf %lf %lf", &nz_h, &sx3, &sy3, &sz3);
    assert(nz_h == 4);
    assert(fabs(sx3) < 1e-6);
    assert(fabs(sy3) < 1e-6);
    assert(fabs(sz3 - 4.0/3.0) < 1e-6);

    assert(fgets(line, sizeof(line), fp) != NULL);
    int Z;
    double charge, ax, ay, az;
    sscanf(line, "%d %lf %lf %lf %lf", &Z, &charge, &ax, &ay, &az);
    assert(Z == 2);
    assert(fabs(charge) < 1e-6);

    int data_lines = 0;
    int total_values = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '\n' || line[0] == '\0') continue;

        char *ptr = line;
        while (*ptr) {
            while (*ptr == ' ' || *ptr == '\t' || *ptr == '\n') ptr++;
            if (*ptr == '\0') break;
            char *end;
            strtod(ptr, &end);
            if (end != ptr) total_values++;
            ptr = end;
        }
        data_lines++;
    }

    int expected_total = 4 * 4 * 4;
    assert(total_values == expected_total);

    int expected_lines = (expected_total + 5) / 6;
    assert(data_lines == expected_lines);

    printf("test_cube_writer: PASS (%d grid values, %d data lines)\n", total_values, data_lines);

    fflush(stderr); fclose(fp);
    remove("test_output.cube");
    free(psi_real);
    free(psi_imag);
    free(psi_abs);
    free(wfn.coeffs);
    free_species_pao(&sys.species[0]);
}

int main(void) {
    test_cube_writer();
    printf("\nAll CUBE writer tests PASSED\n");
    return 0;
}
#endif

// ============================================================
// Main program
// ============================================================

#ifndef TEST_MATH
#ifndef TEST_PAO
#ifndef TEST_DAT
#ifndef TEST_WFN
#ifndef TEST_COMPUTE
#ifndef TEST_CUBE

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <openmx.dat> <wfn.bin> [output.cube]\n", prog);
    fprintf(stderr, "  DFT_DATA path is read from DATA.PATH in openmx.dat\n");
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char *dat_path = argv[1];
    const char *wfn_path = argv[2];
    const char *out_path = argc > 3 ? argv[3] : "wfn.cube";

    System sys;
    memset(&sys, 0, sizeof(sys));

    fprintf(stderr, "[1/4] Parsing %s ...\n", dat_path);
    if (parse_dat(dat_path, &sys) != 0) {
        free_system(&sys);
        return 1;
    }

    fprintf(stderr, "[2/4] Loading %s ...\n", wfn_path);
    Wavefunction wfn;
    memset(&wfn, 0, sizeof(wfn));
    if (load_wavefunction(wfn_path, &sys, &wfn) != 0) {
        free_wavefunction(&wfn);
        free_system(&sys);
        return 1;
    }

    fprintf(stderr, "[3/4] Computing wavefunction (real, imag, density)...\n");
    double *psi_real = NULL, *psi_imag = NULL, *psi_abs = NULL;
    compute_wavefunction(&sys, &wfn, &psi_real, &psi_imag, &psi_abs);
    free_wavefunction(&wfn);
    if (!psi_real || !psi_imag || !psi_abs) {
        free(psi_real); free(psi_imag); free(psi_abs);
        free_system(&sys);
        return 1;
    }


    char base_path[2048];
    strncpy(base_path, out_path, sizeof(base_path) - 1);
    base_path[sizeof(base_path) - 1] = '\0';

    size_t len = strlen(base_path);
    if (len > 5 && strcmp(base_path + len - 5, ".cube") == 0) {
        base_path[len - 5] = '\0';
    }

    char path_real[2048], path_imag[2048], path_abs[2048];
    snprintf(path_real, sizeof(path_real), "%s_real.cube", base_path);
    snprintf(path_imag, sizeof(path_imag), "%s_imag.cube", base_path);
    snprintf(path_abs, sizeof(path_abs), "%s_abs.cube", base_path);

    fprintf(stderr, "[4/4] Writing %s ...\n", path_real);
    if (write_cube(path_real, &sys, psi_real, "Wavefunction real part Re[psi]") != 0) {
        free(psi_real); free(psi_imag); free(psi_abs);
        free_system(&sys);
        return 1;
    }

    fprintf(stderr, "[4/4] Writing %s ...\n", path_imag);
    if (write_cube(path_imag, &sys, psi_imag, "Wavefunction imaginary part Im[psi]") != 0) {
        free(psi_real); free(psi_imag); free(psi_abs);
        free_system(&sys);
        return 1;
    }

    fprintf(stderr, "[4/4] Writing %s ...\n", path_abs);
    if (write_cube(path_abs, &sys, psi_abs, "Wavefunction probability density |psi|^2") != 0) {
        free(psi_real); free(psi_imag); free(psi_abs);
        free_system(&sys);
        return 1;
    }

    free(psi_real);
    free(psi_imag);
    free(psi_abs);
    free_system(&sys);

    fprintf(stderr, "Done. Wrote 3 cube files: %s, %s, %s\n", path_real, path_imag, path_abs);
    return 0;
}

#endif
#endif
#endif
#endif
#endif
#endif
