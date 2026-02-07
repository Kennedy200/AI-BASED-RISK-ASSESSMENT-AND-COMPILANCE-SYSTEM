import styled, { keyframes } from 'styled-components';
import { motion } from 'framer-motion';

// --- Background Animations ---

const float = keyframes`
  0% { transform: translate(0px, 0px) scale(1); }
  33% { transform: translate(30px, -50px) scale(1.1); }
  66% { transform: translate(-20px, 20px) scale(0.9); }
  100% { transform: translate(0px, 0px) scale(1); }
`;

export const HeroSection = styled.section`
  width: 100%;
  min-height: 100vh;
  background: #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 120px 6% 60px;
  position: relative;
  overflow: hidden;
`;

export const BackgroundBlob = styled(motion.div)<{ $color: string; $top: string; $left: string }>`
  position: absolute;
  top: ${({ $top }) => $top};
  left: ${({ $left }) => $left};
  width: 500px;
  height: 500px;
  background: ${({ $color }) => $color};
  filter: blur(80px);
  opacity: 0.4;
  border-radius: 50%;
  z-index: 0;
  animation: ${float} 20s ease-in-out infinite;
`;

// --- Layout ---

export const Container = styled.div`
  max-width: 1200px;
  width: 100%;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 60px;
  align-items: center;
  z-index: 1;

  @media (max-width: 968px) {
    grid-template-columns: 1fr;
    text-align: center;
  }
`;

export const ContentWrapper = styled(motion.div)`
  display: flex;
  flex-direction: column;
  gap: 24px;
  
  @media (max-width: 968px) {
    align-items: center;
  }
`;

export const Badge = styled.div`
  background: white;
  color: var(--color-primary);
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 0.85rem;
  font-weight: 600;
  width: fit-content;
  box-shadow: var(--shadow-sm);
  border: 1px solid #e2e8f0;
  display: flex;
  align-items: center;
  gap: 8px;

  &::before {
    content: '';
    width: 8px;
    height: 8px;
    background: var(--color-primary);
    border-radius: 50%;
  }
`;

export const Headline = styled.h1`
  font-size: 3.5rem;
  line-height: 1.1;
  font-weight: 800;
  color: var(--color-text-main);
  letter-spacing: -1.5px;

  span {
    background: linear-gradient(135deg, var(--color-primary) 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  @media (max-width: 768px) {
    font-size: 2.5rem;
  }
`;

export const Subheadline = styled.p`
  font-size: 1.125rem;
  color: var(--color-text-secondary);
  line-height: 1.6;
  max-width: 500px;
`;

export const ButtonGroup = styled.div`
  display: flex;
  gap: 16px;
  margin-top: 10px;
`;

export const HeroButton = styled(motion.button)`
  padding: 16px 40px;
  border-radius: 12px;
  font-weight: 600;
  font-size: 1.05rem;
  cursor: pointer;
  border: none;
  background: var(--color-primary);
  color: white;
  box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
  transition: all 0.2s;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 25px -5px rgba(37, 99, 235, 0.4);
    background: var(--color-primary-dark);
  }
`;

// --- New Image & Glass Card Styles ---

export const ImageWrapper = styled(motion.div)`
  width: 100%;
  height: 500px;
  position: relative;
  perspective: 1000px;
  
  @media (max-width: 968px) {
    height: 400px;
    margin-top: 40px;
  }
`;

export const HeroImage = styled(motion.img)`
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 24px;
  box-shadow: 
    0 20px 25px -5px rgba(0, 0, 0, 0.1), 
    0 10px 10px -5px rgba(0, 0, 0, 0.04);
  z-index: 1;
`;

export const FloatingCard = styled(motion.div)`
  position: absolute;
  bottom: 40px;
  left: -30px;
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(12px);
  padding: 24px;
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.5);
  box-shadow: 
    0 20px 25px -5px rgba(0, 0, 0, 0.1), 
    0 8px 10px -6px rgba(0, 0, 0, 0.1);
  width: 300px;
  z-index: 2;
  display: flex;
  flex-direction: column;
  gap: 12px;

  @media (max-width: 768px) {
    left: 50%;
    transform: translateX(-50%) !important;
    bottom: -20px;
  }
`;

export const CardHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  
  span:first-child {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--color-text-secondary);
  }
  
  span:last-child {
    background: #fee2e2;
    color: #ef4444;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 4px 8px;
    border-radius: 12px;
  }
`;

export const CardTitle = styled.h4`
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--color-text-main);
  margin: 0;
`;

export const ProgressBar = styled.div`
  width: 100%;
  height: 6px;
  background: #f1f5f9;
  border-radius: 3px;
  overflow: hidden;
  
  div {
    width: 85%;
    height: 100%;
    background: #ef4444;
    border-radius: 3px;
  }
`;

export const CardMeta = styled.div`
  display: flex;
  justify-content: space-between;
  font-size: 0.8rem;
  color: var(--color-text-secondary);
`;