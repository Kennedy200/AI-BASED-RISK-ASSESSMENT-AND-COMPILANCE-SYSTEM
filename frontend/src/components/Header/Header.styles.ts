import styled from 'styled-components';
import { motion } from 'framer-motion';

export const NavContainer = styled(motion.nav)<{ $scrolled: boolean }>`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 80px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 6%;
  z-index: 1000;
  
  /* Modern Glass Effect */
  background: ${({ $scrolled }) => 
    $scrolled ? 'rgba(255, 255, 255, 0.9)' : 'transparent'};
  backdrop-filter: ${({ $scrolled }) => 
    $scrolled ? 'blur(10px)' : 'none'};
  box-shadow: ${({ $scrolled }) => 
    $scrolled ? 'var(--shadow-sm)' : 'none'};
  transition: all 0.3s ease;
`;

export const LogoGroup = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  font-weight: 800;
  font-size: 1.5rem;
  color: var(--color-text-main);
  letter-spacing: -0.5px;
  z-index: 10;
  
  span {
    color: var(--color-primary);
  }
`;

// --- Animated Pill Navigation Styles ---

export const DesktopMenu = styled.div`
  display: flex;
  align-items: center;
  gap: 5px; /* Tight gap for the pill look */
  background: rgba(0, 0, 0, 0.03); /* Subtle container background */
  padding: 5px;
  border-radius: 50px; /* Pill shape */
  border: 1px solid rgba(0,0,0,0.02);

  @media (max-width: 768px) {
    display: none;
  }
`;

export const NavItemWrapper = styled.div`
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
`;

export const HoverPill = styled(motion.div)`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: white;
  border-radius: 30px;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  z-index: 1; /* Behind the text */
`;

export const NavLink = styled.a`
  position: relative;
  font-size: 0.95rem;
  font-weight: 500;
  color: var(--color-text-secondary);
  text-decoration: none;
  padding: 8px 20px;
  border-radius: 30px;
  z-index: 2; /* On top of the pill */
  transition: color 0.2s;
  display: block;

  &:hover {
    color: var(--color-primary);
  }
`;

// --- Auth Buttons ---

export const AuthButtons = styled.div`
  display: flex;
  gap: 16px;
  align-items: center;
`;

export const PrimaryButton = styled(motion.button)`
  background-color: var(--color-primary);
  color: white;
  border: none;
  padding: 10px 24px;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.95rem;
  cursor: pointer;
  box-shadow: var(--shadow-sm);
  transition: background-color 0.2s;

  &:hover {
    background-color: var(--color-primary-dark);
  }
`;