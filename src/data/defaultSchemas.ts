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
    description: "11-point schema for single-side fish morphometric analysis",
    landmarks: [
      { index: 0, name: "Snout Tip", description: "Tip of upper jaw", category: "head" },
      { index: 1, name: "Eye Center", description: "Geometric center of the eye", category: "head" },
      { index: 2, name: "Opercular Edge", description: "Posterior-most bony margin of operculum", category: "head" },
      { index: 3, name: "Pectoral Origin", description: "Anterior attachment of pectoral fin", category: "pectoral-fin" },
      { index: 4, name: "Pelvic Origin", description: "Anterior attachment of pelvic fin", category: "pelvic-fin" },
      { index: 5, name: "Dorsal Origin", description: "Anterior insertion of dorsal fin", category: "dorsal-fin" },
      { index: 6, name: "Dorsal Insertion", description: "Posterior insertion of dorsal fin", category: "dorsal-fin" },
      { index: 7, name: "Anal Origin", description: "Anterior insertion of anal fin", category: "anal-fin" },
      { index: 8, name: "Anal Insertion", description: "Posterior insertion of anal fin", category: "anal-fin" },
      { index: 9, name: "Upper Caudal Peduncle", description: "Dorsal insertion of caudal rays", category: "caudal-fin" },
      { index: 10, name: "Lower Caudal Peduncle", description: "Ventral insertion of caudal rays", category: "caudal-fin" },
    ],
  },
];
