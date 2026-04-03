import { LandmarkSchema } from "../types/Image";

/**
 * Default landmark schemas for common morphometric analyses
 */
export const DEFAULT_SCHEMAS: LandmarkSchema[] = [
  {
    id: "fly-wing",
    name: "Fly Wing (Drosophila) Landmarks",
    description: "12-point schema for Drosophila wing morphometric analysis",
    landmarks: [
      // Margin Intersections (1–6)
      { index: 1,  name: "L5 × Posterior Margin", description: "Intersection of longitudinal vein 5 (L5) and the posterior wing margin", category: "margin" },
      { index: 2,  name: "L4 × Posterior Margin", description: "Intersection of longitudinal vein 4 (L4) and the posterior wing margin", category: "margin" },
      { index: 3,  name: "L3 × Distal Margin",    description: "Intersection of longitudinal vein 3 (L3) and the distal wing margin",   category: "margin" },
      { index: 4,  name: "L2 × Anterior Margin",  description: "Intersection of longitudinal vein 2 (L2) and the anterior wing margin", category: "margin" },
      { index: 5,  name: "L1 × Anterior Margin",  description: "Intersection of longitudinal vein 1 (L1) and the anterior wing margin", category: "margin" },
      { index: 6,  name: "Costal Break",           description: "Notch on the anterior margin at the base of the wing",                 category: "margin" },
      // Crossvein Intersections (7–10)
      { index: 7,  name: "pcv × L5",              description: "Intersection of the posterior crossvein (pcv) and L5",                 category: "crossvein" },
      { index: 8,  name: "pcv × L4",              description: "Intersection of the posterior crossvein (pcv) and L4",                 category: "crossvein" },
      { index: 9,  name: "acv × L4",              description: "Intersection of the anterior crossvein (acv) and L4",                  category: "crossvein" },
      { index: 10, name: "acv × L3",              description: "Intersection of the anterior crossvein (acv) and L3",                  category: "crossvein" },
      // Vein Branching Points (11–12)
      { index: 11, name: "L2/L3 Bifurcation",     description: "Branching point (bifurcation) of veins L2 and L3",                    category: "branch" },
      { index: 12, name: "L4/L5 Bifurcation",     description: "Branching point at the base of veins L4 and L5",                     category: "branch" },
    ],
  },
  {
    id: "fish-morphometrics",
    name: "Fish Lateral Morphometrics",
    description: "11-point schema for single-side fish morphometric analysis",
    landmarks: [
      { index: 1,  name: "Snout Tip",             description: "Tip of upper jaw",                          category: "head" },
      { index: 2,  name: "Dorsal Origin",          description: "Anterior insertion of dorsal fin",          category: "dorsal-fin" },
      { index: 3,  name: "Dorsal Insertion",       description: "Posterior insertion of dorsal fin",         category: "dorsal-fin" },
      { index: 4,  name: "Upper Caudal Peduncle",  description: "Dorsal insertion of caudal rays",           category: "caudal-fin" },
      { index: 5,  name: "Lower Caudal Peduncle",  description: "Ventral insertion of caudal rays",          category: "caudal-fin" },
      { index: 6,  name: "Anal Insertion",         description: "Posterior insertion of anal fin",           category: "anal-fin" },
      { index: 7,  name: "Anal Origin",            description: "Anterior insertion of anal fin",            category: "anal-fin" },
      { index: 8,  name: "Pelvic Origin",          description: "Anterior attachment of pelvic fin",         category: "pelvic-fin" },
      { index: 9,  name: "Opercular Edge",         description: "Posterior-most bony margin of operculum",   category: "head" },
      { index: 10, name: "Pectoral Origin",        description: "Anterior attachment of pectoral fin",       category: "pectoral-fin" },
      { index: 11, name: "Eye Center",             description: "Geometric center of the eye",               category: "head" },
    ],
  },
];
