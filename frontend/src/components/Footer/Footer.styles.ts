import styled from 'styled-components';

export const FooterSection = styled.footer`
  background-color: #0f172a; // Dark Navy
  padding: 80px 6% 40px;
  color: white;
`;

export const FooterGrid = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 1.5fr 1fr 1fr 1fr;
  gap: 60px;
  margin-bottom: 60px;

  @media (max-width: 768px) {
    grid-template-columns: 1fr 1fr;
    gap: 40px;
  }

  @media (max-width: 480px) {
    grid-template-columns: 1fr;
  }
`;

export const FooterBrand = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

export const FooterLogo = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 1.5rem;
  font-weight: 800;
  letter-spacing: -1px;

  span {
    color: var(--color-primary);
  }
`;

export const FooterDesc = styled.p`
  color: #94a3b8;
  font-size: 0.95rem;
  line-height: 1.6;
  max-width: 300px;
`;

export const FooterColumn = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;

  h5 {
    font-size: 1rem;
    font-weight: 700;
    color: white;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
`;

export const FooterLink = styled.a`
  color: #94a3b8;
  text-decoration: none;
  font-size: 0.9rem;
  transition: color 0.2s;

  &:hover {
    color: var(--color-primary);
  }
`;

export const BottomBar = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding-top: 40px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: #64748b;
  font-size: 0.85rem;

  @media (max-width: 768px) {
    flex-direction: column;
    gap: 20px;
    text-align: center;
  }
`;