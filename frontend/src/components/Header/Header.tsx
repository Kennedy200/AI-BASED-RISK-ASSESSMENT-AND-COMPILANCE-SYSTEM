import React, { useState, useEffect } from 'react';
import { FiHexagon } from 'react-icons/fi';
import { AnimatePresence } from 'framer-motion';
import { 
  NavContainer, 
  LogoGroup, 
  DesktopMenu, 
  NavLink, 
  NavItemWrapper,
  HoverPill,
  AuthButtons,
  PrimaryButton 
} from './Header.styles';

const Header: React.FC = () => {
  const [scrolled, setScrolled] = useState(false);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const links = [
    { name: 'Product', href: '#product' },
    { name: 'Solutions', href: '#solutions' },
    { name: 'Resources', href: '#resources' }
  ];

  return (
    <NavContainer $scrolled={scrolled}>
      <LogoGroup>
        <FiHexagon size={24} color="var(--color-primary)" fill="rgba(37, 99, 235, 0.1)" />
        Sentinel<span>AI</span>
      </LogoGroup>

      {/* Sweet Animated Nav */}
      <DesktopMenu>
        {links.map((link, index) => (
          <NavItemWrapper 
            key={link.name}
            onMouseEnter={() => setHoveredIndex(index)}
            onMouseLeave={() => setHoveredIndex(null)}
          >
            <AnimatePresence>
              {hoveredIndex === index && (
                <HoverPill 
                  layoutId="navPill" 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
            </AnimatePresence>
            <NavLink href={link.href}>{link.name}</NavLink>
          </NavItemWrapper>
        ))}
      </DesktopMenu>

      <AuthButtons>
        <PrimaryButton whileHover={{ y: -2 }} whileTap={{ scale: 0.98 }}>
          Log In
        </PrimaryButton>
      </AuthButtons>
    </NavContainer>
  );
};

export default Header;