# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import re
from codecs import open

# if either generated_thy_model.py or generated_thy_parser.py is not available, run 'python -m tatsu -o generated_parser/generated_thy_parser.py -G generated_parser/generated_thy_model.py  thy_model.ebnf'
from database_generation.generated_thy_model import *
from database_generation.generated_thy_parser import ThyParser, KEYWORDS


def xstr(s):
    if s is None:
        return ''
    elif isinstance(s, list):
        return ''.join(map(str, s))
    else:
        return str(s)


def flatten(llist):
    if llist is None:
        return []
    if not isinstance(llist, list):
        return [llist]
    rlist = []
    for a in llist:
        if isinstance(a, list):
            rlist.extend(flatten(a))
        else:
            rlist.append(a)
    return rlist


def to_int_list(list_as_str):
    if list_as_str == '[]':
        return []
    return [int(s) for s in list_as_str[1:-1].split(',')]


def str_of_list(list):
    return str(list).replace(' ', '')


def to_int_tuple(tl_str):
    from ast import literal_eval
    return literal_eval(tl_str)


def str_of_tuple(tup):
    return str(tup).replace(' ', '')


class StringOfTheory(object):
    INDENT = '  '

    def __init__(self, incl_recording=False, incl_evaluation=False):
        self.incl_recording = incl_recording
        self.incl_evaluation = incl_evaluation

    def get_args_str(self, args):
        rr = ''
        for cc in flatten(args):
            if isinstance(cc, FactReference):
                rr += ' ' + self.str_FactReference(cc)
            else:
                assert isinstance(cc, str)
                rr += ' ' + cc
        return rr

    def str_ExtraContext(self, model):
        return '{} {}\n{}\n\n'.format(model.key, ' '.join(model.args), xstr(model.cont))

    def str_ExtraThyCommand(self, model):
        qualifier_str = '' if model.qualifier is None else model.qualifier + ' '
        if model.proof is None:
            proof_str = ''
        else:
            proof_str = '\n' + self.str_ProofBlock(model.proof, 1)
        return qualifier_str + model.key + ' ' + ' '.join(map(str, model.args)) + proof_str + '\n\n'

    def str_Sublocale(self, model):
        return '{} {}\n{}\n\n'.format(model.key, ' '.join(map(str, model.sub_args)), self.str_ProofBlock(model.proof))

    # def str_Interpretation(self,model):
    #     return '{} {}\n{}\n\n'.format(model.key,' '.join(map(str,model.inter_args)),self.str_ProofBlock(model.proof))

    # def str_Lemmas(self, model):
    #     return 'lemmas {}{} = {}\n\n'.format(model.left,xstr(model.modifier),' '.join(map(self.str_FactReference,model.facts)))

    # def str_IsPattern(self,model):
    #     # return '(is {})'.format(xstr(model.cont))
    #     return model.cont

    def str_TextBlock(self, model):
        return model.key + ' ' + xstr(model.opt) + model.cont + '\n\n'

    def str_Definition(self, model):
        qualifier_str = '' if model.qualifier is None else model.qualifier + ' '
        if isinstance(model.vars, Variables):
            return '{}definition{} {} where\n{}\n\n'.format(qualifier_str, xstr(model.locale),
                                                            self.str_Variables(model.vars) \
                                                            , self.str_Propositions(model.props))
        else:
            assert model.vars is None
            return '{}definition{} {}\n\n'.format(qualifier_str, xstr(model.locale), self.str_Propositions(model.props))

    def str_Function(self, model):
        cont_str = self.INDENT
        for cc in flatten(model.cont):
            if cc == '|':
                cont_str += '|\n' + self.INDENT
            else:
                assert isinstance(cc, Propositions)
                cont_str += self.str_Propositions(cc)
        if model.proof is None:
            proof_str = ''
        else:
            proof_str = '\n' + self.str_ProofBlock(model.proof, 1)
        qualifier_str = '' if model.qualifier is None else model.qualifier + ' '
        return '{}{}{} {} where\n{}{}\n\n'.format(qualifier_str, model.key, xstr(model.locale),
                                                  self.str_Variables(model.vars), cont_str, proof_str)

    def str_NamedTheorems(self, model):
        cont_str = ''
        qualifier_str = '' if model.qualifier is None else model.qualifier + ' '
        for cc in model.cont:
            if isinstance(cc, list):
                cont_str += ' ' + ' '.join(cc)
            elif cc == 'and':
                cont_str += '\n' + self.INDENT + 'and '
            else:
                cont_str += cc
        return '{}named_theorems {}\n\n'.format(qualifier_str, cont_str)

    def str_Termination(self, model):
        if model.name is None or model.name in KEYWORDS:
            name_str = ''
        else:
            name_str = model.name
        qualifier_str = '' if model.qualifier is None else model.qualifier + ' '
        return '{}termination{} {}\n{}\n\n'.format(qualifier_str, xstr(model.locale), name_str,
                                                   self.str_ProofBlock(model.proof, 1))

    def str_LocaleClass(self, model):
        prec_or_name_str = self.str_preconditions(model.prec_or_name)
        inherited_names_str = ''
        for c1 in model.inherited_names:
            if isinstance(c1, list):
                assert c1[0] == 'for'
                inherited_names_str += '\n' + self.INDENT + 'for ' + self.str_Variables(c1[1])
            else:
                inherited_names_str += ' ' + c1
        eq_str = '=' if model.eq == '=' else ''

        # inherited_names_str =  ' '.join(map(lambda x: ' '.join(x), model.inherited_names))
        if isinstance(model.ex_thy_stats, list):
            assert model.ex_thy_stats[0] == 'begin' and model.ex_thy_stats[2] == 'end' \
                   and len(model.ex_thy_stats) == 3
            ex_thy_stats_str = ''.join(map(self.str_theory_statement, model.ex_thy_stats[1]))
            return '{} {} {} {}\n{}\nbegin\n\n{}end\n\n'.format(model.key, model.class_name \
                                                                , eq_str, inherited_names_str, prec_or_name_str,
                                                                ex_thy_stats_str)
        else:
            return '{} {} {} {}\n{}\n\n'.format(model.key, model.class_name, eq_str \
                                                , inherited_names_str, prec_or_name_str)

    def str_Context(self, model):
        # print(model.prec_or_name)
        if isinstance(model.prec_or_name, list):
            prec_or_name_str = self.str_preconditions(model.prec_or_name)
        elif model.prec_or_name is None or model.prec_or_name in KEYWORDS:
            prec_or_name_str = ''
        else:
            assert isinstance(model.prec_or_name, str)
            prec_or_name_str = self.INDENT + model.prec_or_name
        thy_stats_str = ''.join(map(self.str_theory_statement, model.thy_stats))
        return 'context {}\nbegin\n\n{}end\n\n'.format(prec_or_name_str, thy_stats_str)

    def str_Instantiations(self, model):
        def str_instantiation(model):
            for_vars_str = '' if model.for_vars is None \
                else ' for {}'.format(self.str_Variables(model.for_vars[1]))
            return '{} = {}{}'.format(xstr(model.left), xstr(model.right), for_vars_str)

        more_insts_str = ''
        for cc in model.more_insts:
            assert cc[0] == 'and' and len(cc) == 2
            more_insts_str += ' and ' + str_instantiation(cc[1])
        return str_instantiation(model.main_inst) + more_insts_str

    def str_Selection(self, model):
        sel_str = '('
        for cc in model.cont:
            if isinstance(cc, list):
                sel_str += ''.join(map(str, cc))
            else:
                sel_str += str(cc)
        return sel_str + ')'

    def str_Propositions(self, model):
        main_name_str = '' if not isinstance(model.main_name, list) else ' '.join(map(str, model.main_name))
        main_conts_str = ' '.join(map(str, model.main_conts))
        more_conts_str = ''
        for cc in model.more_conts:
            cc_str = ' ' + ' '.join(map(str, cc[:-1])) + ' ' + ' '.join(map(xstr, cc[-1]))
            more_conts_str += cc_str
        if isinstance(model.if_if, list):
            assert model.if_if[0] in ['when', 'if']
            assert isinstance(model.if_if[1], Propositions)
            if_if_str = ' ' + model.if_if[0] + ' ' + self.str_Propositions(model.if_if[1])
        else:
            if_if_str = ''
        for_vars_str = '' if model.for_vars is None else ' for ' + self.str_Variables(model.for_vars[1])
        return main_name_str + ' ' + main_conts_str + more_conts_str + if_if_str + for_vars_str

    def str_Variables(self, model):
        main_type_str = xstr(model.main_type)
        more_vars_str = ''
        for cc in model.more_vars:
            if isinstance(cc, list):
                cc_str = ''
                for dd in cc:
                    cc_str += ' ' + xstr(dd)
                more_vars_str += cc_str
                # if cc[0] == 'and':
                #     more_vars_str+=' and ' + xstr(cc[1:])
                # else:
                #     more_vars_str+= ' ' + xstr(cc)
            else:
                more_vars_str += ' ' + xstr(cc)
        return model.main_var + main_type_str + xstr(model.mixfix) + more_vars_str

    def str_Tactic(self, model):
        if model.key == '-':
            return '-'
        elif model.key == '(':
            def str_single_tactic(model):
                if model.key in ['has_type_tac', "has_type_tac'", 'has_type_no_if_tac', 'if_type_tac',
                                 'seq_decl_inv_method', 'seq_stop_inv_method', 'PLM_solver']:
                    return model.key + self.get_args_str(model.tac_args)
                elif model.key == 'sep_auto':
                    return model.key + self.get_args_str(model.method_opt) + self.get_args_str(model.tac_args)
                elif model.key == 'vcg':
                    return 'vcg ' + self.get_args_str(model.vcg_args)
                elif model.key == 'sos':
                    return 'sos ' + self.get_args_str(model.sos_args)
                elif model.key == 'autoref':
                    return 'autoref ' + ' '.join(flatten(model.autoref_args))
                elif model.key == 'r_compose':
                    r_compose_args_str = ''
                    for cc in flatten(model.r_compose_args):
                        if isinstance(cc, FactReference):
                            r_compose_args_str += ' ' + self.str_FactReference(cc)
                        else:
                            assert isinstance(cc, str)
                            r_compose_args_str += ' ' + cc
                    return 'r_compose ' + r_compose_args_str
                elif model.key == 'master_theorem':
                    master_args_str = ''
                    for cc in flatten(model.master_args):
                        if isinstance(cc, FactReference):
                            master_args_str += ' ' + self.str_FactReference(cc)
                        else:
                            assert isinstance(cc, str)
                            master_args_str += ' ' + cc
                    return 'master_theorem ' + master_args_str
                elif model.key in ["transfer'", 'transfer']:
                    transfer_opt_str = ' '.join(flatten(model.transfer_opt))
                    return model.key + ' ' + ' '.join(map(self.str_FactReference, model.facts)) + ' ' + transfer_opt_str
                elif model.key in ['cases', 'relation', 'case_tac', 'injectivity_solver', 'transfer_hma',
                                   'inst_existentials']:
                    cases_trm_str = ' '.join(map(str, model.cases_trm))
                    # cases_trm_str = xstr(model.cases_trm) if model.cases_trm != 'rule' else '' # TatSu bug
                    for_vars_str = '' if model.for_vars is None else \
                        ' for {}'.format(self.str_Variables(model.for_vars[1]))
                    set_part_str = ' ' + ' '.join(flatten(model.set_part)) if isinstance(model.set_part, list) else ''
                    rule_fact_str = ' rule:' + self.str_FactReference(model.rule_fact[-1]) if isinstance(
                        model.rule_fact, list) else ''
                    return model.key + ' '.join(
                        model.tac_opt) + ' ' + cases_trm_str + for_vars_str + set_part_str + rule_fact_str
                elif model.key == 'subst':
                    subst_opt_str = '' if not model.subst_opt else ' ' + ' '.join(model.subst_opt)
                    return model.key + subst_opt_str + ' ' + ' '.join(map(self.str_FactReference, model.facts))
                elif model.key == 'rewrite':
                    return 'rewrite ' + ' '.join(flatten(model.rewrite_args)) + ' ' + self.str_FactReference(model.fact)
                elif model.key in ['rule_tac', 'drule_tac', 'erule_tac', 'frule_tac', 'cut_tac', 'subst_tac',
                                   'hoare_rule']:
                    inst_str = self.str_Instantiations(model.inst[0]) + ' in' if isinstance(model.inst, list) else ''
                    facts_str = ' '.join(map(self.str_FactReference, model.facts))
                    return model.key + ' ' + ' '.join(model.tac_opt) + inst_str + ' ' + facts_str
                elif model.key in ['frpar', 'frpar2']:
                    return model.key + ' ' + ' '.join(flatten(model.frpar_args))
                elif model.key in ['approximation']:
                    return model.key + ' ' + ' '.join(flatten(model.approx_args))
                elif model.key in ['rename_tac', 'tactic', 'subgoal_tac', 'thin_tac', 'ind_cases', 'rotate_tac']:
                    for_vars_str = '' if model.for_vars is None else \
                        ' for {}'.format(self.str_Variables(model.for_vars[1]))
                    return '{} {}{}{}'.format(model.key, ' '.join(model.tac_opt), ' '.join(model.rename_trm),
                                              for_vars_str)
                elif model.key == 'use':
                    facts_str = ' '.join(map(self.str_FactReference, model.facts))
                    if isinstance(model.more_tac, list):
                        assert len(model.more_tac) == 3
                        singles_str = ''
                        for cc in model.more_tac[1]:
                            if cc in [',', ';', '|']:
                                singles_str += cc + ' '
                            else:  # cc is a single tactic
                                singles_str += str_single_tactic(cc)
                        return 'use {} in \<open>{}\<close>'.format(facts_str, singles_str)
                    else:
                        assert isinstance(model.more_tac, str)
                        return 'use {} in {}'.format(facts_str, model.more_tac)
                elif model.key == 'all':
                    if isinstance(model.more_tac, list):
                        assert len(model.more_tac) == 3
                        singles_str = ''
                        for cc in model.more_tac[1]:
                            if cc in [',', ';', '|']:
                                singles_str += cc + ' '
                            else:  # cc is a single tactic
                                singles_str += str_single_tactic(cc)
                        return 'all \<open>{}\<close>'.format(singles_str)
                    else:
                        assert isinstance(model.more_tac, str)
                        return 'all {}'.format(model.more_tac)
                elif model.key in ['induct_tac', 'induct', 'induction', 'coinduct', 'coinduction', 'nominal_induct']:
                    induct_arb_str = ' ' + ' '.join(flatten(model.induct_arb))
                    if isinstance(model.induct_rule, list):
                        induct_rule_str = ' rule: ' + ' '.join(map(self.str_FactReference, model.induct_rule[-1]))
                    else:
                        induct_rule_str = ''
                    # to counter a TatSu bug
                    induct_trm_str = ' '.join(flatten(model.induct_trm))
                    # induct_trm_str = '' if model.induct_trm is None or model.induct_trm \
                    #                         in ['rule','arbitrary','set'] else ' '.join(model.induct_trm)
                    return model.key + ' ' + induct_trm_str + induct_arb_str + induct_rule_str
                elif isinstance(model.key, Tactic):
                    return self.str_Tactic(model.key)
                else:
                    assert model.key is None, model.key
                    assert model.method_name is not None, model.method_name
                    method_opt_str = '' if not model.method_opt else ' ' + ' '.join(model.method_opt)
                    attributed_facts_str = ''
                    for af in flatten(model.attributed_facts):
                        if isinstance(af, FactReference):
                            attributed_facts_str += ' ' + self.str_FactReference(af)
                        else:
                            assert isinstance(af, str)
                            attributed_facts_str += ' ' + af

                    # for af in model.attributed_facts:
                    #     if not af:
                    #         attributed_facts_str += ''
                    #     elif isinstance(af[-1],list):
                    #         attributed_facts_str +=' '+' '.join(af[:-1])\
                    #             + ' '.join(map(self.str_FactReference,af[-1]))
                    #     else:
                    #         assert isinstance(af[-1], FactReference)
                    #         attributed_facts_str +=' '+ ' '.join(map(self.str_FactReference,af))
                    return model.method_name + method_opt_str + attributed_facts_str

            more_tactics_str = ''
            for mt in model.more_tactics:
                assert mt[0] in [',', ';', '|']
                more_tactics_str += ' ' + mt[0] + ' '
                more_tactics_str += str_single_tactic(mt[1])
            return '(' + str_single_tactic(model.main_tactic) + more_tactics_str + ')' + xstr(model.tac_mod)
        else:
            # model.key is a name
            return model.key + xstr(model.tac_mod)

    def str_FactReference(self, model):
        forward_modifier_str = '' if not isinstance(model.for_mod, ForwardModifier) else self.str_ForwardModifier(
            model.for_mod)
        sel_str = '' if not model.sel else self.str_Selection(
            model.sel)  # 'model.sel is not None' -> 'not model.sel' due to a parser bug when model.sel=[]
        if model.fact_name is not None:
            return str(model.fact_name) + sel_str + forward_modifier_str
        else:
            assert model.cont is not None
            return model.cont + forward_modifier_str

    def str_ForwardModifier(self, model):
        cont_str = ''
        for sf in model.cont:
            if sf == ',':
                cont_str += sf + ' '
            else:
                # now with the rule single_forward
                if sf.key in ['OF', 'THEN', 'folded', 'unfolded', 'simplified', 'case_product', 'to_pred', 'to_set',
                              'FCOMP']:
                    cont_str += sf.key + ' '
                    cont_str += '' if sf.opt is None else ' ' + sf.opt
                    cont_str += ' '.join(map(lambda x: '_' if x == '_' else self.str_FactReference(x), sf.facts))
                elif sf.key == 'of':
                    cont_str += sf.key + ' ' + ' '.join(map(xstr, sf.of_arg))
                elif sf.key in ['rotated', 'consumes']:
                    cont_str += sf.key + ' ' + xstr(sf.rotated_arg)
                elif sf.key in ['case_names', 'case_conclusion']:
                    cont_str += sf.key + ' ' + ' '.join(sf.case_args)
                elif sf.key == 'internalize_sort':
                    cont_str += sf.key + ' ' + ' '.join(flatten(sf.internalize_sort_args))
                elif sf.key == 'where':
                    cont_str += sf.key + ' ' + self.str_Instantiations(sf.where_arg)
                else:
                    # assert sf.key in ['rule_format','symmetric','abs_def'\
                    #     ,'elim_format',['simp','add'],['simp','del'],'simp',['intro', '!'],'intro','to_set','pred_set_conv','mono_set','arith',['transfer_rule','del'],'transfer_rule','tendsto_intros','Transfer.transferred','measurable','measurable_dest','cong','derivative_intros',
                    #     ['dest','!'],'dest','trans'], sf.key
                    cont_str += ' '.join(flatten(sf.key))
        return '[' + cont_str + ']'

    # def str_LongName(self,model):
    #     name_str = ''
    #     for cc in model.prefix_names:
    #         name_str+=''.join(map(str,cc) )
    #     name_str=name_str+ str(model.main_name)
    #     if model.quot is None:
    #         return name_str
    #     else:
    #         return '\"'+name_str+'\"'

    def str_DiagStatement(self, model):
        if model.key == 'record_facts':
            return 'record_facts ( {} ) {} {}'.format(' '.join(model.attrs) \
                                                      , self.str_FactReference(model.fact_ref), model.txt)
        elif model.key == 'record_all_facts':
            return 'record_all_facts ( {} )'.format(' '.join(model.attrs))
        elif model.key == 'check_derivation':
            # print(model.fact_asms[0])
            # print(model.fact_asms[0].sel)
            # # print(model.fact_asms[0].sel is None)
            # print( json.dumps(model.fact_ref.asjson(),indent=4) )
            # print(type(model.fact_asms[0]))
            # print(self.str_FactReference(model.fact_asms[0]) )
            # print(list(map(self.str_FactReference,model.fact_asms)))
            fact_asms_str = ' '.join(list(map(self.str_FactReference, model.fact_asms)))
            # print(fact_asms_str)
            return 'check_derivation ( {} ) {} {} ( {} )'.format(' '.join(model.attrs) \
                                                                 , model.raw_seq,
                                                                 self.str_FactReference(model.fact_ref), fact_asms_str)
        elif model.key == 'check_derivation_C':
            fact_asms_str = ' '.join(list(map(self.str_FactReference, model.fact_asms)))
            return 'check_derivation_C ( {} ) {} {} {} ( {} )'.format(' '.join(model.attrs) \
                                                                      , model.raw_seq,
                                                                      self.str_FactReference(model.fact_ref),
                                                                      self.str_FactReference(model.fact_ref2),
                                                                      fact_asms_str)
        else:
            raise Exception('Unknown key', model.key)

    def str_ListOfDiagStatement(self, list_of_model, indent_level=0):
        indent_str = self.INDENT * indent_level

        if (self.incl_recording or self.incl_evaluation) and list_of_model:
            diag_str = ('\n' + indent_str).join(map(self.str_DiagStatement, list_of_model))
            diag_str = indent_str + diag_str + '\n'
        else:
            diag_str = ''
        return diag_str

    def str_ProofQed(self, model, indent_level=0):
        indent_str = self.INDENT * indent_level
        # The following isinstance judgement is to counter a possible bug where main_tac_str has been assigned a string (e.g. 'fix' or 'assume') rather than a Tactic object
        main_tac_str = '' if not isinstance(model.main_tac, Tactic) else self.str_Tactic(model.main_tac)
        closing_tac_str = '' if not isinstance(model.closing_tac, Tactic) else self.str_Tactic(model.closing_tac)
        isar_stats_str = ''
        for cc in model.isar_stats:
            if isinstance(cc, IsarStatement):
                isar_stats_str += self.str_IsarStatement(cc, indent_level + 1)
        return '{}proof {}\n{}{}qed {}'.format(
            indent_str \
            , main_tac_str \
            # ,''.join(map(lambda m: self.str_IsarStatement(m,indent_level+1),model.isar_stats))\
            , isar_stats_str \
            , indent_str, closing_tac_str)

    def str_RefinementStep(self, model, indent_level=0):
        indent_str = self.INDENT * indent_level
        diag_str_pre = self.str_ListOfDiagStatement(model.diag_stats_pre, indent_level)
        if model.key == 'using' or model.key == 'unfolding':
            facts_str = ''
            for cc in model.facts:
                if isinstance(cc, FactReference):
                    facts_str += ' ' + self.str_FactReference(cc)
                else:
                    facts_str += ' ' + cc
            return '{}{}{}{}\n'.format(diag_str_pre, indent_str, model.key, facts_str)
        elif model.key == 'including':
            return '{}including {}\n'.format(diag_str_pre, ' '.join(model.incl_args))
        elif model.key in ['apply', 'applyS', 'applyF', 'apply1']:
            return '{}{}{} {}\n'.format(diag_str_pre, indent_str, model.key, self.str_Tactic(model.tac))
        elif model.key in ['focus']:
            if model.tac is None or model.tac in KEYWORDS:
                return '{}{}{} \n'.format(diag_str_pre, indent_str, model.key)
            else:
                return '{}{}{} {}\n'.format(diag_str_pre, indent_str, model.key, self.str_Tactic(model.tac))
        elif model.key == 'supply':
            # sup_opt_str = ' '.join(flatten(model.sup_opt))
            # facts_str = ' '.join(map(self.str_FactReference,model.facts))
            supply_args_str = ''
            for cc in flatten(model.supply_args):
                if isinstance(cc, FactReference):
                    supply_args_str += ' ' + self.str_FactReference(cc)
                else:
                    supply_args_str += ' ' + str(cc)
            return '{}{}{}{}\n'.format(diag_str_pre, indent_str, model.key, supply_args_str)
        elif model.key in ['defer', 'prefer', 'back', 'solved']:
            return '{}{} {}\n'.format(diag_str_pre, model.key, xstr(model.tac_arg))
        else:
            assert model.key is None and model.sblock is not None
            return self.str_SubgoalBlock(model.sblock, indent_level)

    def str_ClosingStep(self, model, indent_level=0):
        indent_str = self.INDENT * indent_level
        diag_str_pre = self.str_ListOfDiagStatement(model.diag_stats_pre, indent_level)
        if model.key == 'by':
            closing_tac_str = '' if not isinstance(model.closing_tac, Tactic) else ' ' + self.str_Tactic(
                model.closing_tac)
            return '{}{}by {}{}\n'.format(diag_str_pre, indent_str, self.str_Tactic(model.main_tac), closing_tac_str)
        else:
            assert model.key in ['done', '..', '.', 'sorry', 'oops', '\<proof>']
            return diag_str_pre + indent_str + model.key + '\n'

    def str_SubgoalBlock(self, model, indent_level=0):
        indent_str = self.INDENT * indent_level
        if model.bname is None:
            bname_str = ''
        elif model.bname in KEYWORDS:  # to counter a TatSu bug, where model.bname be wrongly assigned the next token
            bname_str = ''
        else:
            bname_str = ' ' + model.bname
        # prems_str = '' if model.prems is None else ' premises {}'.format(model.prems[1])
        prems_str = ' ' + ' '.join(flatten(model.prems))
        for_vars_str = '' if model.for_vars is None else \
            ' for {}'.format(self.str_Variables(model.for_vars[1]))

        # rsteps_str = ''
        # for cc in model.rsteps:
        #     if isinstance(cc,RefinementStep):
        #         rsteps_str+=self.str_RefinementStep(cc,indent_level+1)
        # if isinstance(model.cstep,ProofQed):
        #     cstep_str = self.str_ProofQed(model.cstep,indent_level+1)
        # else:
        #     assert isinstance(model.cstep,ClosingStep)
        #     cstep_str = self.str_ClosingStep(model.cstep,indent_level+1)
        # return '{}subgoal{}{}{}\n{}{}'.format(indent_str,bname_str,prems_str,for_vars_str,rsteps_str,cstep_str)
        return '{}subgoal{}{}{}\n{}'.format(indent_str, bname_str, prems_str \
                                            , for_vars_str, self.str_ProofBlock(model.proof, indent_level))

    def str_ProofBlock(self, model, indent_level=0):
        if model == '@phantom':
            model = ProofBlock(key='@phantom', diag_stats_pre=[], diag_stats_post=[])
        if not isinstance(model, ProofBlock):  # to counter a TatSu bug
            return ''

        diag_str_post = self.str_ListOfDiagStatement(model.diag_stats_post, indent_level)
        diag_str_pre = self.str_ListOfDiagStatement(model.diag_stats_pre, indent_level)
        # for st in model.diag_stats:
        #     diag_str+= indent_str+self.str_DiagStatement(st)+'\n'

        if isinstance(model.key, ProofQed):
            # rsteps_str = ''.join(map(lambda m: self.str_RefinementStep(m,indent_level+1),model.rsteps))
            rsteps_str = ''
            for cc in model.rsteps:
                if isinstance(cc, RefinementStep):
                    rsteps_str += self.str_RefinementStep(cc, indent_level + 1)
            return '{}{}{}\n{}'.format(
                diag_str_pre \
                , rsteps_str \
                , self.str_ProofQed(model.key, indent_level)
                , diag_str_post)
        elif isinstance(model.key, ClosingStep):
            assert model.rsteps is not None
            # rsteps_str = ''.join(map(lambda m: self.str_RefinementStep(m,indent_level+1),model.rsteps))
            rsteps_str = ''
            for cc in model.rsteps:
                if isinstance(cc, RefinementStep):
                    rsteps_str += self.str_RefinementStep(cc, indent_level + 1)
            key_str = self.str_ClosingStep(model.key, indent_level + 1)
            return '{}{}{}{}'.format(diag_str_pre, rsteps_str, key_str, diag_str_post)
        elif model.key == '@phantom':
            return '{}{}@phantom\n{}'.format(diag_str_pre, self.INDENT * (indent_level + 1), diag_str_post)
        else:
            raise Exception('Unknown key', model.key)

    def str_IsarStatement(self, model, indent_level=0):
        def str_pre_facts():
            if not model.pre_facts:
                return ''
            pf_str = indent
            for cc in flatten(model.pre_facts):
                if isinstance(cc, FactReference):
                    pf_str += self.str_FactReference(cc) + ' '
                else:
                    assert isinstance(cc, str)
                    pf_str += cc + ' '
            return pf_str + '\n'

        indent = self.INDENT * indent_level
        if model.key == '{':
            isar_stats_str = ''
            for cc in model.isar_stats:
                if isinstance(cc, IsarStatement):
                    isar_stats_str += self.str_IsarStatement(cc, indent_level + 1)
            # isar_stats_str = ''.join(map(lambda m: self.str_IsarStatement(m,indent_level+1),model.isar_stats))
            diag_str = self.str_ListOfDiagStatement(model.diag_stats_post, indent_level)
            return str_pre_facts() + indent + '{\n' + isar_stats_str + indent + '}\n' + diag_str
        elif model.key in ['assume', 'presume']:
            # return 'assume'+self.str_Propositions(model.props)
            diag_str = self.str_ListOfDiagStatement(model.diag_stats_post, indent_level)
            return str_pre_facts() + indent + model.key + ' ' + self.str_Propositions(model.props) + '\n' + diag_str
            # return '{}{} {}\n{}'.format(str_pre_facts(), model.key,self.str_Propositions(model.props),diag_str)
        elif model.key == 'case':
            diag_str = self.str_ListOfDiagStatement(model.diag_stats_post, indent_level)
            case_name_str = ' ' + ' '.join(flatten(model.case_name)) if model.case_name is not None else ''
            return '{}{}case{} {}\n{}'.format(str_pre_facts(), indent, case_name_str, model.case_arg, diag_str)
        elif model.key == 'define':
            diag_str = self.str_ListOfDiagStatement(model.diag_stats_post, indent_level)
            return '{}{}define {} where {}\n{}'.format(str_pre_facts(), indent, self.str_Variables(model.vars) \
                                                       , self.str_Propositions(model.props), diag_str)
        elif model.key == 'note':
            diag_str1 = self.str_ListOfDiagStatement(model.diag_stats_pre1, indent_level)
            diag_str2 = self.str_ListOfDiagStatement(model.diag_stats_pre2, indent_level)
            # name_eq_str = ' '.join(map(str,model.name_eq)) if isinstance(model.name_eq,list) else ''
            # facts_str = ' '.join(map(self.str_FactReference,model.facts))
            name_eq_str = ''
            for cc in flatten(model.name_eq):
                if isinstance(cc, FactReference):
                    name_eq_str += ' ' + self.str_FactReference(cc)
                else:
                    name_eq_str += ' ' + str(cc)
            return '{}{}{}{}note{}\n{}'.format(diag_str1, str_pre_facts(), diag_str2, indent, name_eq_str,
                                               self.str_ProofBlock(model.proof, indent_level))
        elif model.key in ['have', 'show', 'thus', 'hence']:
            if isinstance(model.props, Propositions):
                props_str = self.str_Propositions(model.props)
            else:
                assert False  # is this going to happen?
                props_str = ''.join(model.props)
            if isinstance(model.when_if, list):
                assert model.when_if[0] in ['when']
                assert isinstance(model.when_if[1], Propositions)
                when_if_str = ' ' + model.when_if[0] + ' ' + self.str_Propositions(model.when_if[1])
            else:
                when_if_str = ''
            for_vars_str = '' if model.for_vars is None else ' for ' + self.str_Variables(model.for_vars[1])
            # diag_str=' '.join(map(self.str_DiagStatement,model.diag_stats)) + ' ' if model.diag_stats else ''
            diag_str1 = self.str_ListOfDiagStatement(model.diag_stats_pre1, indent_level)
            diag_str2 = self.str_ListOfDiagStatement(model.diag_stats_pre2, indent_level)
            return '{}{}{}{}{} {}{}{}\n{}'.format(diag_str1, str_pre_facts() \
                                                  , diag_str2, indent, model.key, props_str, when_if_str, for_vars_str \
                                                  , self.str_ProofBlock(model.proof, indent_level))
        elif model.key == 'interpret':
            for_vars_str = '' if model.for_vars is None else ' for ' + self.str_Variables(model.for_vars[1])
            # diag_str=' '.join(map(self.str_DiagStatement,model.diag_stats)) + ' ' if model.diag_stats else ''
            diag_str1 = self.str_ListOfDiagStatement(model.diag_stats_pre1, indent_level)
            diag_str2 = self.str_ListOfDiagStatement(model.diag_stats_pre2, indent_level)
            return '{}{}{}{}interpret {}{}\n{}'.format(diag_str1, str_pre_facts(), diag_str2, indent \
                                                       , ' '.join(map(str, model.inter_args)), for_vars_str \
                                                       , self.str_ProofBlock(model.proof, indent_level))
        elif model.key == 'obtain':
            props_str = self.str_Propositions(model.props)
            # diag_str=' '.join(map(self.str_DiagStatement,model.diag_stats)) + ' ' if model.diag_stats else ''
            diag_str1 = self.str_ListOfDiagStatement(model.diag_stats_pre1, indent_level)
            diag_str2 = self.str_ListOfDiagStatement(model.diag_stats_pre2, indent_level)
            if model.vars is None:
                return '{}{}{}{}obtain {}\n{}'.format(diag_str1, str_pre_facts() \
                                                      , diag_str2, indent, props_str,
                                                      self.str_ProofBlock(model.proof, indent_level))
            else:
                vars_str = self.str_Variables(model.vars)
                return '{}{}{}{}obtain {} where {}\n{}'.format(diag_str1, str_pre_facts() \
                                                               , diag_str2, indent, vars_str, props_str,
                                                               self.str_ProofBlock(model.proof, indent_level))
        elif model.key == 'consider':
            consider_args_str = ' '.join(model.consider_args)
            diag_str1 = self.str_ListOfDiagStatement(model.diag_stats_pre1, indent_level)
            diag_str2 = self.str_ListOfDiagStatement(model.diag_stats_pre2, indent_level)
            return '{}{}{}{}consider {}\n{}'.format(diag_str1, str_pre_facts() \
                                                    , diag_str2, indent, consider_args_str,
                                                    self.str_ProofBlock(model.proof, indent_level))
        elif model.key == 'guess':
            diag_str1 = self.str_ListOfDiagStatement(model.diag_stats_pre1, indent_level)
            diag_str2 = self.str_ListOfDiagStatement(model.diag_stats_pre2, indent_level)
            return '{}{}{}{}guess {}\n{}'.format(diag_str1, str_pre_facts() \
                                                 , diag_str2, indent, self.str_Variables(model.vars),
                                                 self.str_ProofBlock(model.proof, indent_level))
        elif model.key == 'fix':
            vars_str = self.str_Variables(model.vars)
            # diag_str=self.str_ListOfDiagStatement(model.diag_stats,indent_level)
            # diag_str=' '.join(map(self.str_DiagStatement,model.diag_stats)) + ' ' if model.diag_stats else ''
            return '{}{}fix {}\n'.format(str_pre_facts(), indent, vars_str)
        elif model.key == 'write':
            # diag_str=self.str_ListOfDiagStatement(model.diag_stats,indent_level)
            return '{}write {}\n'.format(indent, ' '.join(model.write_args))
        elif model.key == 'let':
            # diag_str=self.str_ListOfDiagStatement(model.diag_stats,indent_level)
            # diag_str=' '.join(map(self.str_DiagStatement,model.diag_stats)) + ' ' if model.diag_stats else ''
            return '{}{}let {}\n'.format(str_pre_facts(), indent, self.str_Instantiations(model.inst))
        elif model.key == 'next':
            # diag_str=self.str_ListOfDiagStatement(model.diag_stats,indent_level)
            # diag_str=' '.join(map(self.str_DiagStatement,model.diag_stats)) + ' ' if model.diag_stats else ''
            return '{}next\n'.format(self.INDENT * (indent_level - 1))
        elif model.key == 'include':
            # diag_str=self.str_ListOfDiagStatement(model.diag_stats,indent_level)
            return '{}include {}\n'.format(indent, ' '.join(model.incl_args))
        else:
            raise Exception('Unknown key', model.key)

    def str_preconditions(self, model):
        assert isinstance(model, list)
        preconditions_str = ''
        for pp in model:
            assert len(pp) == 2
            if pp[0] in ['assumes', 'defines']:
                preconditions_str += self.INDENT + pp[0] + ' ' + self.str_Propositions(pp[1]) + '\n'
            elif pp[0] in ['fixes', 'constrains']:
                preconditions_str += self.INDENT + pp[0] + ' ' + self.str_Variables(pp[1]) + '\n'
            elif pp[0] in ['includes', 'notes']:
                assert isinstance(pp[1], list)
                preconditions_str += self.INDENT + pp[0] + ' ' + ' '.join(pp[1])
            else:
                raise Exception('Unknown precondition', pp[0])
        return preconditions_str

    def str_TheoremStatement(self, model):
        diag_str = self.str_ListOfDiagStatement(model.diag_stats, 1)
        if model.key == 'shows':
            preconditions_str = self.str_preconditions(model.preconditions)
            # preconditions_str=''
            # for pp in model.preconditions:
            #     if pp[0] == 'assumes':
            #         preconditions_str+=self.INDENT +'assumes '+self.str_Propositions(pp[1]) + '\n'
            #     elif pp[0] == 'fixes':
            #         preconditions_str+=self.INDENT +'fixes '+self.str_Variables(pp[1]) + '\n'
            #     else:
            #         raise Exception('Unknown precondition', pp[0])
            return '{}{}shows {}\n{}'.format(preconditions_str, self.INDENT \
                                             , self.str_Propositions(model.props), diag_str)
        elif model.key == 'obtains':
            preconditions_str = self.str_preconditions(model.preconditions)
            # preconditions_str=''
            # for pp in model.preconditions:
            #     if pp[0] == 'assumes':
            #         preconditions_str+=self.INDENT +'assumes '+self.str_Propositions(pp[1]) + '\n'
            #     elif pp[0] == 'fixes':
            #         preconditions_str+=self.INDENT +'fixes '+self.str_Variables(pp[1]) + '\n'
            #     else:
            #         raise Exception('Unknown precondition', pp[0])
            ob_cont_str = ''
            for cc in flatten(model.ob_cont):
                if isinstance(cc, Variables):
                    ob_cont_str += ' ' + self.str_Variables(cc)
                elif isinstance(cc, Propositions):
                    ob_cont_str += ' ' + self.str_Propositions(cc)
                else:
                    assert isinstance(cc, str)
                    ob_cont_str += ' ' + cc
            return '{}{}obtains{}\n{}'.format(preconditions_str, self.INDENT, ob_cont_str, diag_str)
        else:
            assert model.key is None
            return self.str_Propositions(model.props) + '\n' + diag_str

    def str_Theorem(self, model):
        qualifier_str = '' if model.qualifier is None else model.qualifier + ' '

        # thm_name_str = '' if model.thm_name is None or model.thm_name in KEYWORDS else model.thm_name
        thm_name_args_str = ' '.join(flatten(model.thm_name_args))
        diag_str = self.str_ListOfDiagStatement(model.diag_stats, 0)
        return qualifier_str + model.key + ' ' + xstr(
            model.locale) + thm_name_args_str + '\n' + self.str_TheoremStatement(model.thm_stat) + self.str_ProofBlock(
            model.proof, 0) + diag_str + '\n\n'

    def str_Theorem_noproof(self, model):
        qualifier_str = '' if model.qualifier is None else model.qualifier + ' '

        # thm_name_str = '' if model.thm_name is None or model.thm_name in KEYWORDS else model.thm_name
        thm_name_args_str = ' '.join(flatten(model.thm_name_args))
        diag_str = self.str_ListOfDiagStatement(model.diag_stats, 0)
        return qualifier_str + model.key + ' ' + xstr(
            model.locale) + thm_name_args_str + '\n' + self.str_TheoremStatement(model.thm_stat) + diag_str + '\n\n'

    def str_theory_statement(self, model):
        if isinstance(model, TextBlock):
            # return self.str_TextBlock(model) #TODO REVERT THIS
            # return 'Andrzej klamie ################################################################3'
            return ''
        elif isinstance(model, Theorem):
            # return self.str_Theorem(model) TODO REVERT THIS
            return self.str_Theorem_noproof(model)
        elif isinstance(model, Definition):
            return self.str_Definition(model)
        elif isinstance(model, Function):
            return self.str_Function(model)
        elif isinstance(model, Termination):
            return self.str_Termination(model)
        elif isinstance(model, LocaleClass):
            return self.str_LocaleClass(model)
        elif isinstance(model, ExtraThyCommand):
            return self.str_ExtraThyCommand(model)
        # elif isinstance(model,Lemmas):
        #     return self.str_Lemmas(model)
        elif isinstance(model, NamedTheorems):
            return self.str_NamedTheorems(model)
        elif isinstance(model, Context):
            return self.str_Context(model)
        elif isinstance(model, Sublocale):
            return self.str_Sublocale(model)
        # elif isinstance(model,Interpretation):
        #     return self.str_Interpretation(model)
        elif isinstance(model, ExtraContext):
            return self.str_ExtraContext(model)
        else:
            raise Exception('Unknown theory_statement', model)

    def str_Theory(self, model):
        thy_stats_str = ''.join(map(self.str_theory_statement, model.thy_stats))
        imported_thy_names_str = self.INDENT + ' '.join(model.imported_thy_names)
        if self.incl_recording and hasattr(model, 'diag_thy_path_recording'):
            imported_thy_names_str += ' ' + model.diag_thy_path_recording
        if self.incl_evaluation and hasattr(model, 'diag_thy_path_evaluation'):
            # imported_thy_names_str+=' '+model.diag_thy_path_evaluation
            pass

        # if self.incl_diag and hasattr(model,'diag_thy_path'):
        #     paths = [s for s in model.diag_thy_path if not s.endswith('Recording/Sequence_Evaluation\"')] if not self.incl_evaluation else model.diag_thy_path
        #     imported_thy_names_str+=' '+' '.join(paths)
        # if isinstance(model.keywords,list):
        #     keywords_str = 'keywords\n'
        #     for cc in model.keywords[1]:
        #         if isinstance(cc,list):
        #             assert len(cc) == 3
        #             keywords_str+='{}{} :: {}'.format(self.INDENT,' '.join(cc[0]), cc[2])
        #         else:
        #             assert cc=='and'
        #             keywords_str+=' and\n'
        # else:
        #     keywords_str = ''
        keywords_str = ''
        for cc in flatten(model.keywords):
            if cc == 'keywords':
                keywords_str += self.INDENT + 'keywords\n' + self.INDENT * 2
            elif cc == 'and':
                keywords_str += 'and\n' + self.INDENT * 2
            else:
                keywords_str += cc + ' '

        if isinstance(model.abbrevs, list):
            abbrevs_str = ' '.join(flatten(model.abbrevs))
        else:
            abbrevs_str = ''

        thy_name_str = model.thy_name if model.thy_name[0] == '"' and model.thy_name[-1] == '"' \
            else '"' + model.thy_name + '"'

        return '{}theory {} imports\n{}\n{}\n{}\nbegin\n\n{}end\n'.format( \
            ''.join(map(self.str_TextBlock, model.text_blocks)) \
            , thy_name_str, imported_thy_names_str, keywords_str, abbrevs_str
            , thy_stats_str)


def get_block_isar(model, isar,
                   comp_idx):  # if the block is None (e.g. when encountering IsarStatement 'assume') IsarStatement is returned instead
    def is_when_if(model):
        if model.key in {'have', 'show', 'thus', 'hence'}:
            if isinstance(model.when_if, list) or isinstance(model.for_vars, list):
                return True
        if hasattr(model, 'props') and isinstance(model.props, Propositions):
            if hasattr(model.props, 'if_if') and isinstance(model.props.if_if, list):
                return True
            if hasattr(model.props, 'for_vars') and isinstance(model.props.for_vars, list):
                return True
        return False

    def is_pre_isar(isar_model, comp_idx):
        if comp_idx is None:
            return False
        pre_facts, _, _, _ = isar_model.proof.retrieve_facts(isar_model)
        if comp_idx < len(pre_facts):
            return True
        else:
            return False

    if isar.proof is None:
        return ('POST', isar)
    elif is_pre_isar(isar, comp_idx):
        return ('PRE1', isar)
    elif is_when_if(isar):
        return ('PRE', isar.proof)
    else:
        return ('POST', isar.proof)


tt = "have  \"(P1 x \<and> P2 x) \<longleftrightarrow> (P1' \<and> P2')\" if H : \"x \<sqsubset> z\" for x\
  using less_trans[OF H zz1] less_trans[OF H zz2] z1 zz1 z2 zz2\
    by auto"


def test():
    thy_src = open('/home/szymon/Downloads/afp-2021-12-14/thys/Concurrent_Ref_Alg/Refinement_Lattice.thy').read()
    parser = ThyParser(semantics=ThyModelBuilderSemantics())
    model = parser.parse(thy_src)

    boi = StringOfTheory(True).str_Theory(model)
    print(boi)


def iterate_over_dir_family(dir_path):
    # make directory tree for outlines
    from pathlib import Path
    import os
    print("PATH", os.path.join(dir_path, "thys"))

    names = glob.glob(os.path.join(dir_path, 'thys/*'))
    for name in names:
        if os.path.isdir(name):
            filename = name.replace("thys", "outlines", 1)
            Path(filename).mkdir(parents=True, exist_ok=True)

    # parse, generate outline and write to the outline directory tree
    # start with list of theory files
    names = glob.glob(os.path.join(dir_path, 'thys/*/*.thy'))
    files_attempted, failed_attempts = 0, 0
    failed_filenames = ""
    for i, name in enumerate(names):
        print(name)
        files_attempted += 1
        print(i)

        # Magic starts here
        thy_src = open(name).read()
        parser = ThyParser(semantics=ThyModelBuilderSemantics())

        try:
            model = parser.parse(thy_src)
            boi = StringOfTheory(True).str_Theory(model)
            # quick and dirty cleaner of proofs from locales and other weird objects that aren't lemmas and theorems
            boi = re.sub("(\n *)+\n", "▒", boi)
            boi = re.sub("\nproof[^▒]+\nqed", "", boi)
            boi = re.sub("(▒ *)+", "\n\n", boi)
        except Exception as e:
            failed_attempts += 1
            failed_filenames += "\n" + name
            continue
        filename = name[:-4] + '_outline.txt'
        filename = filename.replace("thys", "outlines", 1)
        with open(filename, "w") as f:
            f.write(boi)

    # write failed files
    filename = os.path.join(dir_path, "failed_outlines.txt")
    with open(filename, "w") as f:
        # first percentage of files failed, then their paths
        f.write(str(100 * failed_attempts / files_attempted) + " % \n" + failed_filenames)

    print("################################################### DONE")

def get_outline(file_name):
    thy_src = open(file_name).read()
    parser = ThyParser(semantics=ThyModelBuilderSemantics())

    model = parser.parse(thy_src)
    boi = StringOfTheory(True).str_Theory(model)
    # quick and dirty cleaner of proofs from locales and other weird objects that aren't lemmas and theorems
    boi = re.sub("(\n *)+\n", "▒", boi)
    boi = re.sub("\nproof[^▒]+\nqed", "", boi)
    boi = re.sub("(▒ *)+", "\n\n", boi)

    return boi

def get_theorems(file_name):
    outline_string = get_outline(file_name)

    theorems = {"unnamed": []}
    # the keywords meaning lemma
    keywords = ["lemma", "theorem", "corollary", "proposition", "schematic_goal"]
    # split to blocks for each lemma
    blocks = outline_string.split("\n\n")
    blocks = [i.lstrip() for i in blocks]
    i = 0
    for block in blocks:
        # remove attributes
        block_clean = re.sub('\[.+?\]', '', block)
        # get and clean first line to get name of lemma
        first_line = block_clean.split("\n")[0].strip().replace(":", '')
        # split for individ words
        first_line_split = first_line.split()
        # check if first line is interesting for us
        if first_line_split[0] in keywords:
            key = first_line_split[1] if len(first_line_split) > 1 else "unnamed"
            # case of super short lemmas, get everything after name from first line
            if len(first_line_split) > 2:
                value = first_line.split(key, 1)[1].split("by")[0].strip()
            # normal lemmas, get every other line
            else:
                value = block_clean.split("\n", 1)[1].strip()
            if key == "unnamed":
                theorems[key] = theorems[key] + [(value, i)]
            else:
                theorems[key] = (value, i)
            i += 1
    return theorems