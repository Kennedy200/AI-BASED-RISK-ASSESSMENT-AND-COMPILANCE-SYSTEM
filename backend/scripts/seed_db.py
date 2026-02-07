"""
Script to seed database with initial data.
"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import AsyncSession
from app.db.base import init_db, SyncSessionLocal
from app.models.user import User, Role, UserStatus, user_roles
from app.core.security import hash_password


async def seed_roles(db: AsyncSession):
    """Create default roles."""
    from app.core.security.permissions import ROLE_PERMISSIONS
    
    roles = []
    for role_name, permissions in ROLE_PERMISSIONS.items():
        role = Role(
            name=role_name,
            description=f"{role_name.replace('_', ' ').title()} role",
            permissions=",".join(permissions) if isinstance(permissions, list) else ""
        )
        roles.append(role)
        db.add(role)
    
    await db.flush()
    return roles


async def seed_admin_user(db: AsyncSession):
    """Create admin user."""
    # Create super admin
    admin = User(
        email="admin@fraudetect.com",
        username="admin",
        hashed_password=hash_password("admin123"),
        first_name="System",
        last_name="Administrator",
        status=UserStatus.ACTIVE,
        is_active=True,
        is_superuser=True
    )
    db.add(admin)
    await db.flush()
    
    # Assign all roles to admin
    result = await db.execute(select(Role))
    roles = result.scalars().all()
    admin.roles = roles
    
    await db.flush()
    return admin


async def main():
    """Seed database."""
    print("Seeding database...")
    
    # Initialize database
    await init_db()
    
    from sqlalchemy import select
    from app.db.base import AsyncSessionLocal
    
    async with AsyncSessionLocal() as db:
        # Check if already seeded
        result = await db.execute(select(User))
        if result.scalar_one_or_none():
            print("Database already seeded.")
            return
        
        # Seed roles
        print("Creating roles...")
        await seed_roles(db)
        
        # Seed admin user
        print("Creating admin user...")
        await seed_admin_user(db)
        
        await db.commit()
    
    print("Database seeded successfully!")
    print("\nDefault credentials:")
    print("  Username: admin")
    print("  Password: admin123")


if __name__ == "__main__":
    asyncio.run(main())
