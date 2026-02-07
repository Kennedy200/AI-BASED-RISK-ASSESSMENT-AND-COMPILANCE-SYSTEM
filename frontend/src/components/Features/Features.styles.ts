import styled from 'styled-components';
import { motion } from 'framer-motion';

export const Section = styled.section`
  padding: 100px 6%;
  background-color: #f8fafc; // Soft slate gray background
  display: flex;
  flex-direction: column;
  align-items: center;
`;

export const HeaderArea = styled.div`
  text-align: center;
  max-width: 700px;
  margin-bottom: 60px;
`;

export const Title = styled.h2`
  font-size: 2.5rem;
  font-weight: 800;
  color: var(--color-text-main);
  letter-spacing: -1px;
  margin-bottom: 16px;

  span {
    color: var(--color-primary);
  }
`;

export const Subtitle = styled.p`
  font-size: 1.1rem;
  color: var(--color-text-secondary);
  line-height: 1.6;
`;

export const Grid = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 30px;
  max-width: 1200px;
  width: 100%;

  @media (max-width: 968px) {
    grid-template-columns: 1fr;
  }
`;

export const FeatureCard = styled(motion.div)`
  background: white;
  padding: 40px;
  border-radius: 24px;
  border: 1px solid #e2e8f0;
  transition: all 0.3s ease;
  display: flex;
  flex-direction: column;
  gap: 20px;

  &:hover {
    border-color: var(--color-primary);
    box-shadow: var(--shadow-lg);
    transform: translateY(-5px);
  }
`;

export const IconWrapper = styled.div`
  width: 50px;
  height: 50px;
  background: rgba(37, 99, 235, 0.1);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-primary);
`;

export const FeatureTitle = styled.h3`
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--color-text-main);
`;

export const FeatureDescription = styled.p`
  font-size: 1rem;
  color: var(--color-text-secondary);
  line-height: 1.6;
`;