import styled from 'styled-components';
import { motion } from 'framer-motion';

export const CTASection = styled.section`
  padding: 100px 6%;
  background-color: white;
  display: flex;
  justify-content: center;
`;

export const CTAContainer = styled(motion.div)`
  width: 100%;
  max-width: 1200px;
  background: var(--color-primary);
  border-radius: 32px;
  padding: 80px 40px;
  text-align: center;
  color: white;
  position: relative;
  overflow: hidden;
  box-shadow: 0 20px 40px rgba(37, 99, 235, 0.2);
`;

export const CTATitle = styled.h2`
  font-size: 3rem;
  font-weight: 800;
  margin-bottom: 24px;
  letter-spacing: -1px;
`;

export const CTASubtitle = styled.p`
  font-size: 1.25rem;
  opacity: 0.9;
  margin-bottom: 40px;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
`;

export const CTAButton = styled(motion.button)`
  background: white;
  color: var(--color-primary);
  border: none;
  padding: 16px 40px;
  border-radius: 12px;
  font-size: 1.1rem;
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 10px 20px rgba(0,0,0,0.1);
`;