"""add section_id and section_target to chat_sessions

Revision ID: b2c3d4e5f6a7
Revises: 3c493736e444
Create Date: 2026-04-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = '3c493736e444'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'chat_sessions',
        sa.Column('section_id', sa.String(36), nullable=True),
    )
    op.add_column(
        'chat_sessions',
        sa.Column('section_target', sa.String(100), nullable=True),
    )
    op.create_index(
        'ix_chat_sessions_section',
        'chat_sessions',
        ['project_id', 'section_id'],
    )


def downgrade() -> None:
    op.drop_index('ix_chat_sessions_section', table_name='chat_sessions')
    op.drop_column('chat_sessions', 'section_target')
    op.drop_column('chat_sessions', 'section_id')
