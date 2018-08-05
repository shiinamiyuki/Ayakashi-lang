from abc import abstractmethod


class Visitor:
    def __init__(self):
        pass

    @abstractmethod
    def visit_cast_expr(self, node):
        pass

    @abstractmethod
    def visit_if_stmt(self, node):
        pass

    @abstractmethod
    def visit_while_stmt(self, node):
        pass

    @abstractmethod
    def visit_binary_expr(self, node):
        pass

    @abstractmethod
    def visit_number(self, node):
        pass

    @abstractmethod
    def visit_identifier(self, node):
        pass

    @abstractmethod
    def visit_unary_expr(self, node):
        pass

    @abstractmethod
    def visit_declaration(self, node):
        pass

    @abstractmethod
    def visit_block(self, node):
        pass

    @abstractmethod
    def visit_func_def(self, node):
        pass

    @abstractmethod
    def visit_func_def_arg(self, node):
        pass

    @abstractmethod
    def visit_index(self, node):
        pass

    @abstractmethod
    def visit_call(self, node):
        pass

    @abstractmethod
    def visit_call_arg(self, node):
        pass

    @abstractmethod
    def visit_string(self, node):
        pass

    @abstractmethod
    def visit_return(self, node):
        pass

    @abstractmethod
    def visit_struct(self, node):
        pass

    @abstractmethod
    def visit_ref_type(self, node):
        pass

    @abstractmethod
    def visit_ptr_type(self, node):
        pass

    @abstractmethod
    def visit_c_header(self, node):
        pass

    @abstractmethod
    def visit_c_definition(self, node):
        pass

    @abstractmethod
    def visit_c_type(self, node):
        pass

    @abstractmethod
    def visit_import(self, node):
        pass

    @abstractmethod
    def visit_implementation(self, node):
        pass

    @abstractmethod
    def visit_method_def(self, node):
        pass

    @abstractmethod
    def visit_interface(self, node):
        pass

    @abstractmethod
    def visit_impl_for(self, node):
        pass

    @abstractmethod
    def visit_generic(self, node):
        pass

    @abstractmethod
    def visit_type_inference(self, node):
        pass