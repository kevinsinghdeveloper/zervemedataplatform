from zervedataplatform.connectors.sql_connectors.PostgresSqlConnector import PostgresSqlConnector


class SqlConnectorHandler:
    @staticmethod
    def get_sql_connector(db_config):
        db_type = db_config['database_type']
        if db_type.lower() == 'postgres':
            return PostgresSqlConnector(db_config)
        else:
            raise Exception(f"Unknown database type: {db_type}")

