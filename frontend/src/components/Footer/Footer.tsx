import React from 'react';
import { FiHexagon, FiGithub, FiTwitter, FiLinkedin } from 'react-icons/fi';
import { 
  FooterSection, 
  FooterGrid, 
  FooterBrand, 
  FooterLogo, 
  FooterDesc, 
  FooterColumn, 
  FooterLink, 
  BottomBar 
} from './Footer.styles';

const Footer: React.FC = () => {
  return (
    <FooterSection>
      <FooterGrid>
        <FooterBrand>
          <FooterLogo>
            <FiHexagon size={24} color="var(--color-primary)" fill="rgba(37, 99, 235, 0.2)" />
            Sentinel<span>AI</span>
          </FooterLogo>
          <FooterDesc>
            Next-generation fraud detection and compliance monitoring for the modern financial sector. 
            Built with explainable AI at its core.
          </FooterDesc>
          <div style={{ display: 'flex', gap: '20px' }}>
            <FooterLink href="#"><FiTwitter size={20} /></FooterLink>
            <FooterLink href="#"><FiLinkedin size={20} /></FooterLink>
            <FooterLink href="#"><FiGithub size={20} /></FooterLink>
          </div>
        </FooterBrand>

        <FooterColumn>
          <h5>Product</h5>
          <FooterLink href="#features">Features</FooterLink>
          <FooterLink href="#performance">Performance</FooterLink>
          <FooterLink href="#process">Workflow</FooterLink>
        </FooterColumn>

        <FooterColumn>
          <h5>Company</h5>
          <FooterLink href="#">About Us</FooterLink>
          <FooterLink href="#">Research</FooterLink>
          <FooterLink href="#">Contact</FooterLink>
        </FooterColumn>

        <FooterColumn>
          <h5>Legal</h5>
          <FooterLink href="#">Privacy Policy</FooterLink>
          <FooterLink href="#">Terms of Service</FooterLink>
          <FooterLink href="#">Compliance Docs</FooterLink>
        </FooterColumn>
      </FooterGrid>

      <BottomBar>
        <p>© 2026 Sentinel AI Project. Final Year Academic Submission.</p>
        <p>Terms And Services</p>
      </BottomBar>
    </FooterSection>
  );
};

export default Footer;