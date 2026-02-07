import styled from 'styled-components';
import { motion } from 'framer-motion';

export const Section = styled.section`
  padding: 100px 6%;
  background-color: #f0f7ff; // Very light blue tint
  display: flex;
  flex-direction: column;
  align-items: center;
`;

export const Container = styled.div`
  max-width: 1200px;
  width: 100%;
  display: grid;
  grid-template-columns: 1fr 1.5fr;
  gap: 60px;
  align-items: start;

  @media (max-width: 968px) {
    grid-template-columns: 1fr;
  }
`;

export const MetricsColumn = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

export const MetricCard = styled(motion.div)`
  background: white;
  padding: 24px;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  box-shadow: var(--shadow-sm);
`;

export const MetricHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;

  h4 {
    font-weight: 700;
    color: var(--color-text-main);
  }

  span {
    font-family: var(--font-mono);
    font-weight: 800;
    color: var(--color-primary);
  }
`;

export const ProgressBarBase = styled.div`
  width: 100%;
  height: 8px;
  background: #f1f5f9;
  border-radius: 4px;
  overflow: hidden;
`;

export const ProgressBarFill = styled(motion.div)<{ $width: string }>`
  height: 100%;
  width: ${({ $width }) => $width};
  background: var(--color-primary);
  border-radius: 4px;
`;

export const TableColumn = styled(motion.div)`
  background: white;
  border-radius: 24px;
  border: 1px solid #e2e8f0;
  padding: 40px;
  box-shadow: var(--shadow-lg);
`;

export const ComparisonTable = styled.table`
  width: 100%;
  border-collapse: collapse;
  text-align: left;

  th {
    padding-bottom: 20px;
    font-size: 0.85rem;
    text-transform: uppercase;
    color: var(--color-text-secondary);
    letter-spacing: 1px;
    border-bottom: 1px solid #f1f5f9;
  }

  td {
    padding: 20px 0;
    border-bottom: 1px solid #f8fafc;
    font-weight: 500;
    color: var(--color-text-main);

    &:first-child {
      font-weight: 700;
      color: var(--color-primary);
    }
  }
`;