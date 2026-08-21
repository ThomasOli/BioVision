import { Variants, Transition } from "framer-motion"

// Spring transition for natural feel
const springTransition: Transition = {
  type: "spring",
  stiffness: 400,
  damping: 30,
}

// Smooth ease transition
const smoothTransition: Transition = {
  type: "tween",
  ease: [0.25, 0.1, 0.25, 1],
  duration: 0.2,
}

export const modalContent: Variants = {
  initial: {
    opacity: 0,
    scale: 0.96,
    y: 10,
  },
  animate: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: springTransition,
  },
  exit: {
    opacity: 0,
    scale: 0.96,
    y: 10,
    transition: { duration: 0.15 },
  },
}

// Button micro-interactions
export const buttonHover = {
  whileHover: { scale: 1.02 },
  transition: { duration: 0.15 },
}

export const buttonTap = {
  whileTap: { scale: 0.98 },
}

// Card hover effect
export const cardHover: Variants = {
  initial: {},
  hover: {
    y: -3,
    boxShadow: "0 10px 32px rgba(0, 0, 0, 0.10), 0 2px 8px rgba(0, 0, 0, 0.06)",
    transition: { duration: 0.18, ease: [0.22, 1, 0.36, 1] },
  },
}

// Stagger animations for lists
export const staggerContainer: Variants = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.05,
      delayChildren: 0.1,
    },
  },
  exit: {
    transition: {
      staggerChildren: 0.03,
      staggerDirection: -1,
    },
  },
}

export const staggerItem: Variants = {
  initial: { opacity: 0, y: 8 },
  animate: {
    opacity: 1,
    y: 0,
    transition: smoothTransition,
  },
  exit: {
    opacity: 0,
    y: 8,
    transition: { duration: 0.15 },
  },
}

// Dropzone drag feedback
export const dropzoneActive: Variants = {
  initial: {
    borderColor: "hsl(var(--border))",
    backgroundColor: "hsl(var(--muted))",
  },
  active: {
    borderColor: "hsl(var(--primary))",
    backgroundColor: "hsl(var(--primary) / 0.05)",
    transition: { duration: 0.15 },
  },
}

// Sidebar stagger animation
export const sidebarContainer: Variants = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.06,
      delayChildren: 0.15,
    },
  },
}

export const sidebarItem: Variants = {
  initial: { opacity: 0, x: -10 },
  animate: {
    opacity: 1,
    x: 0,
    transition: smoothTransition,
  },
}

