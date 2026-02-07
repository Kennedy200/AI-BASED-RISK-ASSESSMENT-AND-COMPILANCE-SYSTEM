import styled from 'styled-components';
import { motion } from 'framer-motion';

export const ProcessSection = styled.section`
  padding: 100px 6%;
  background: white;
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
`;

export const Container = styled.div`
  max-width: 1000px;
  width: 100%;
  position: relative;
  margin-top: 60px;
`;

// The vertical line that connects steps
export const TimelineLine = styled.div`
  position: absolute;
  left: 24px;
  top: 0;
  bottom: 0;
  width: 2px;
  background: #f1f5f9;
  
  @media (max-width: 768px) {
    left: 15px;
  }
`;

// The animated part of the line that fills on scroll
export const ActiveLine = styled(motion.div)`
  position: absolute;
  left: 0;
  top: 0;
  width: 2px;
  background: var(--color-primary);
  transform-origin: top;
`;

export const StepRow = styled(motion.div)`
  display: flex;
  gap: 40px;
  margin-bottom: 80px;
  position: relative;
  
  &:last-child {
    margin-bottom: 0;
  }

  @media (max-width: 768px) {
    gap: 20px;
  }
`;

export const StepNumber = styled.div`
  width: 50px;
  height: 50px;
  background: white;
  border: 2px solid #f1f5f9;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  color: #cbd5e1;
  z-index: 2;
  transition: all 0.3s ease;
  flex-shrink: 0;

  &.active {
    border-color: var(--color-primary);
    color: var(--color-primary);
    box-shadow: 0 0 15px rgba(37, 99, 235, 0.2);
  }

  @media (max-width: 768px) {
    width: 32px;
    height: 32px;
    font-size: 0.8rem;
  }
`;

export const StepContent = styled.div`
  padding-top: 10px;
  flex: 1;
`;

export const StepTitle = styled.h3`
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--color-text-main);
  margin-bottom: 12px;
`;

export const StepDescription = styled.p`
  font-size: 1.1rem;
  color: var(--color-text-secondary);
  line-height: 1.6;
  max-width: 600px;
`;

// --- NEW SLEEK DESIGN BOX ---
export const StepImage = styled(motion.div)`
  margin-top: 30px;
  width: 100%;
  max-width: 500px;
  height: 240px;
  background: #f8fafc;
  border-radius: 20px;
  border: 1px solid #e2e8f0; // Solid border instead of dashed
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
  box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
`;

// --- STEP 1: INGESTION VISUALS ---
export const FileIcon = styled(motion.div)<{ $color: string }>`
  width: 60px;
  height: 80px;
  background: white;
  border-radius: 8px;
  border: 2px solid ${({ $color }) => $color};
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  box-shadow: var(--shadow-md);
  position: absolute;

  &::before {
    content: '';
    width: 30%;
    height: 4px;
    background: #e2e8f0;
    border-radius: 2px;
  }
`;

// --- STEP 2: ANALYSIS VISUALS ---
export const ModelChip = styled(motion.div)`
  padding: 8px 16px;
  background: white;
  border-radius: 30px;
  border: 1px solid #e2e8f0;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  font-weight: 700;
  color: var(--color-primary);
  box-shadow: var(--shadow-sm);
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;

  span {
    width: 8px;
    height: 8px;
    background: var(--color-primary);
    border-radius: 50%;
  }
`;

// --- STEP 3: INTELLIGENCE VISUALS ---
export const ShapBar = styled(motion.div)<{ $width: string; $positive: boolean }>`
  height: 12px;
  width: ${({ $width }) => $width};
  background: ${({ $positive }) => $positive ? '#ef4444' : '#22c55e'};
  border-radius: 6px;
  margin-bottom: 8px;
`;