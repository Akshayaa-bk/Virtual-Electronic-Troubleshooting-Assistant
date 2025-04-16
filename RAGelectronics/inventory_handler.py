import re
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ----- SQLAlchemy Base Class and Model -----

Base = declarative_base()

class InventoryItem(Base):
    """
    SQLAlchemy model representing a row in the 'inventory' table.
    Stores product and spare part information along with availability.
    """
    __tablename__ = 'inventory'
    
    id = Column(Integer, primary_key=True)
    product_name = Column(String(255))
    spare_part_name = Column(String(255))
    description = Column(Text)
    quantity_available = Column(Integer)


# ----- DATABASE SETUP -----

def setup_database():
    """
    Establish connection to the SQLite database and create tables if missing.
    
    Returns:
        engine (SQLAlchemy Engine): Database engine
        Session (sessionmaker): Factory for new session objects
    """
    engine = create_engine('sqlite:///inventory.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session


# ----- EXCEL IMPORT FUNCTION -----

def import_inventory_from_excel(excel_file, engine, Session, clear_inventory=False):
    """
    Load inventory data from an Excel file and insert it into the database.
    
    Args:
        excel_file (str): Path to the Excel file
        engine: SQLAlchemy engine instance
        Session: SQLAlchemy session maker
        clear_inventory (bool): If True, clear existing inventory before import
    """
    df = pd.read_excel(excel_file)
    session = Session()

    if clear_inventory:
        session.query(InventoryItem).delete()
        session.commit()

    for _, row in df.iterrows():
        item = InventoryItem(
            product_name=row.get('Product Name', ''),
            spare_part_name=row.get('Spare Part Name', ''),
            description=row.get('Description', ''),
            quantity_available=row.get('Quantity Available', 0)
        )
        session.add(item)

    session.commit()
    session.close()


# ----- INVENTORY QUERY FUNCTIONS -----

def search_inventory(search_term, Session):
    """
    Search inventory records based on general terms or special keywords.
    
    Args:
        search_term (str): User-provided search term
        Session: SQLAlchemy session maker
        
    Returns:
        list: InventoryItem objects matching the criteria
    """
    session = Session()

    if search_term.lower() == "low_stock":
        results = session.query(InventoryItem).filter(
            InventoryItem.quantity_available > 0,
            InventoryItem.quantity_available <= 10
        ).all()

    elif search_term.lower() == "out_of_stock":
        results = session.query(InventoryItem).filter(
            InventoryItem.quantity_available == 0
        ).all()

    else:
        results = session.query(InventoryItem).filter(
            (InventoryItem.product_name.ilike(f'%{search_term}%')) |
            (InventoryItem.spare_part_name.ilike(f'%{search_term}%')) |
            (InventoryItem.description.ilike(f'%{search_term}%'))
        ).all()

    session.close()
    return results


def get_part_inventory(part_name, Session):
    """
    Fetch inventory entries for a specific part name.
    
    Args:
        part_name (str): Spare part name to search
        Session: SQLAlchemy session maker
        
    Returns:
        list: InventoryItem objects
    """
    session = Session()
    results = session.query(InventoryItem).filter(
        InventoryItem.spare_part_name.ilike(f'%{part_name}%')
    ).all()
    session.close()
    return results


# ----- INVENTORY MANAGEMENT FUNCTIONS -----

def update_inventory_quantity(item_id, new_quantity, Session):
    """
    Update the available quantity of a specific inventory item.
    
    Args:
        item_id (int): Inventory item ID
        new_quantity (int): New quantity to set
        Session: SQLAlchemy session maker
        
    Returns:
        bool: True if successful, False otherwise
    """
    session = Session()
    item = session.query(InventoryItem).filter(InventoryItem.id == item_id).first()

    if item:
        item.quantity_available = new_quantity
        session.commit()
        session.close()
        return True

    session.close()
    return False


def get_inventory_stats(Session):
    """
    Calculate summary statistics for the inventory.
    
    Args:
        Session: SQLAlchemy session maker
        
    Returns:
        dict: Statistics including totals and stock status
    """
    session = Session()

    total_items = session.query(InventoryItem).count()
    low_stock = session.query(InventoryItem).filter(
        InventoryItem.quantity_available > 0,
        InventoryItem.quantity_available <= 10
    ).count()
    out_of_stock = session.query(InventoryItem).filter(
        InventoryItem.quantity_available == 0
    ).count()

    session.close()
    
    return {
        "total_items": total_items,
        "low_stock": low_stock,
        "out_of_stock": out_of_stock
    }


# ----- QUERY ANALYSIS FUNCTIONS -----

def is_replacement_query(message):
    """
    Check whether the user query relates to part replacement.
    
    Args:
        message (str): Input query text
        
    Returns:
        bool: True if replacement-related keywords are found
    """
    replacement_keywords = [
        "replace", "replacement", "substitute", "swap", "change",
        "new part", "spare part", "fix", "broken", "damaged",
        "not working", "faulty", "inventory", "stock", "available"
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in replacement_keywords)


def extract_part_names(query, Session):
    """
    Identify spare part names mentioned in the user query.
    
    Args:
        query (str): User's message text
        Session: SQLAlchemy session maker
        
    Returns:
        list: Matched part names found in the query
    """
    session = Session()
    parts = session.query(InventoryItem.spare_part_name).all()
    all_parts = {part[0].lower() for part in parts if part[0]}

    query_lower = query.lower()
    found_parts = [part for part in all_parts if part in query_lower]

    session.close()
    return found_parts
