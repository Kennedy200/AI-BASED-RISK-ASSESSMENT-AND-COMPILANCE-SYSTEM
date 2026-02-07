import styled from 'styled-components';
import { motion } from 'framer-motion';

export const Section = styled.section`
  padding: 100px 6%;
  background-color: white;
  display: flex;
  flex-direction: column;
  align-items: center;
`;

export const AccordionContainer = styled.div`
  max-width: 800px;
  width: 100%;
  margin-top: 40px;
`;

export const AccordionItem = styled.div`
  border-bottom: 1px solid #e2e8f0;
  padding: 24px 0;
`;

export const Question = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  
  h4 {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--color-text-main);
  }

  span {
    color: var(--color-primary);
    font-size: 1.5rem;
  }
`;

export const Answer = styled(motion.div)`
  overflow: hidden;
  color: var(--color-text-secondary);
  font-size: 1rem;
  line-height: 1.6;
`;