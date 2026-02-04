import { LandmarkSchema } from "../types/Image";

/**
 * Default landmark schemas for common morphometric analyses
 */
export const DEFAULT_SCHEMAS: LandmarkSchema[] = [
  {
    id: "butterfly-wing",
    name: "Butterfly Wing Morphometrics",
    description: "Standard 18-point schema for butterfly wing analysis",
    landmarks: [
      { index: 0, name: "Left Forewing Apex", description: "Tip of left forewing", category: "forewing" },
      { index: 1, name: "Right Forewing Apex", description: "Tip of right forewing", category: "forewing" },
      { index: 2, name: "Left Forewing Base", description: "Attachment point of left forewing", category: "forewing" },
      { index: 3, name: "Right Forewing Base", description: "Attachment point of right forewing", category: "forewing" },
      { index: 4, name: "Left Hindwing Apex", description: "Tip of left hindwing", category: "hindwing" },
      { index: 5, name: "Right Hindwing Apex", description: "Tip of right hindwing", category: "hindwing" },
      { index: 6, name: "Left Hindwing Base", description: "Attachment point of left hindwing", category: "hindwing" },
      { index: 7, name: "Right Hindwing Base", description: "Attachment point of right hindwing", category: "hindwing" },
      { index: 8, name: "Left Wing Vein R1", description: "Radial vein 1 on left wing", category: "veins" },
      { index: 9, name: "Right Wing Vein R1", description: "Radial vein 1 on right wing", category: "veins" },
      { index: 10, name: "Left Wing Vein M1", description: "Medial vein 1 on left wing", category: "veins" },
      { index: 11, name: "Right Wing Vein M1", description: "Medial vein 1 on right wing", category: "veins" },
      { index: 12, name: "Left Wing Vein Cu1", description: "Cubital vein 1 on left wing", category: "veins" },
      { index: 13, name: "Right Wing Vein Cu1", description: "Cubital vein 1 on right wing", category: "veins" },
      { index: 14, name: "Body Anterior", description: "Head attachment point", category: "body" },
      { index: 15, name: "Body Posterior", description: "Abdomen tip", category: "body" },
      { index: 16, name: "Left Antenna Tip", description: "Tip of left antenna", category: "antennae" },
      { index: 17, name: "Right Antenna Tip", description: "Tip of right antenna", category: "antennae" },
    ],
  },
  {
    id: "fish-morphometrics",
    name: "Fish Lateral Morphometrics",
    description: "23-point schema for single-side fish morphometric analysis",
    landmarks: [
      // Head region
      { index: 0, name: "Tip of Snout", description: "Anterior-most point of the snout", category: "head" },
      { index: 1, name: "Tip of Maxillary", description: "Posterior tip of the maxillary bone", category: "head" },
      { index: 2, name: "Pre-orbital", description: "Anterior margin of the eye orbit", category: "head" },
      { index: 3, name: "Post-orbital", description: "Posterior margin of the eye orbit", category: "head" },
      { index: 4, name: "Forehead", description: "Dorsal point of the head above the eye", category: "head" },
      { index: 5, name: "Branchiostegals", description: "Base of gill opening (branchiostegal membrane)", category: "head" },
      // First dorsal fin
      { index: 6, name: "First Dorsal Fin Insertion", description: "Insertion point of first dorsal fin", category: "dorsal-fin" },
      { index: 7, name: "First Dorsal Fin Spine Tip", description: "Tip of first spiny ray of first dorsal fin", category: "dorsal-fin" },
      { index: 8, name: "Last Spiny Ray Insertion (D1)", description: "Insertion of last spiny ray of first dorsal fin", category: "dorsal-fin" },
      // Pectoral fin
      { index: 9, name: "Pectoral Fin Insertion", description: "Insertion point of pectoral fin on body", category: "pectoral-fin" },
      { index: 10, name: "Pectoral Fin Tip", description: "Distal tip of pectoral fin", category: "pectoral-fin" },
      // Pelvic fin
      { index: 11, name: "Pelvic Fin Insertion", description: "Insertion point of pelvic fin on body", category: "pelvic-fin" },
      { index: 12, name: "Pelvic Fin Tip", description: "Distal tip of pelvic fin", category: "pelvic-fin" },
      // Second dorsal fin
      { index: 13, name: "Second Dorsal Fin Insertion", description: "Insertion point of second dorsal fin", category: "dorsal-fin" },
      { index: 14, name: "Second Dorsal Fin Spine Tip", description: "Tip of first spiny ray of second dorsal fin", category: "dorsal-fin" },
      { index: 15, name: "Last Soft Ray Insertion (D2)", description: "Insertion of last soft ray of second dorsal fin", category: "dorsal-fin" },
      // Anal fin
      { index: 16, name: "Anal Fin Insertion", description: "Insertion point of anal fin", category: "anal-fin" },
      { index: 17, name: "Anal Fin Spine Tip", description: "Tip of first spiny ray of anal fin", category: "anal-fin" },
      { index: 18, name: "Last Soft Ray Insertion (A)", description: "Insertion of last soft ray of anal fin", category: "anal-fin" },
      // Caudal fin
      { index: 19, name: "First Caudal Ray Insertion", description: "Insertion of first (upper) caudal fin ray", category: "caudal-fin" },
      { index: 20, name: "Last Caudal Ray Insertion", description: "Insertion of last (lower) caudal fin ray", category: "caudal-fin" },
      { index: 21, name: "Upper Caudal Fin Tip", description: "Upper tip of caudal fin", category: "caudal-fin" },
      { index: 22, name: "Lower Caudal Fin Tip", description: "Lower tip of caudal fin", category: "caudal-fin" },
    ],
  },
];
